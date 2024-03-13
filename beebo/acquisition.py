from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from typing import Union, Optional, Tuple
import torch
from gpytorch import settings
import numpy as np
from copy import deepcopy
from .utils.cholesky_inference import GPPosteriorPredictor
from linear_operator.utils.cholesky import psd_safe_cholesky
from enum import Enum

class LogDetMethod(Enum):
    """Used to specify the method for computing the log determinant of the covariance matrices."""
    SVD = "svd"
    CHOLESKY = "cholesky"
    TORCH = "torch"

class AugmentedPosteriorMethod(Enum):
    """Used to specify the method for augmenting the model with new points and computing the posterior covariance."""
    NAIVE = "naive"
    CHOLESKY = "cholesky"
    GET_FANTASY_MODEL = "get_fantasy_model" # NOTE this isn't memory safe yet

class BatchedEnergyEntropyBO(AnalyticAcquisitionFunction):
    r"""The BEE-BO batch acquisition function. Jointly optimizes a batch of points by minimizing

    Args:
        model: A fitted single-outcome GP model (must be in batch mode if
            candidate sets X will be)
        temperature: A scalar representing the temperature. 
            higher temperature -> more exploration
        kernel_amplitude: The amplitude of the kernel. Defaults to 1.0.
            This is used to bring the temperature to a scale that is comparable to 
            UCB's hyperparameter `beta`.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.
        logdet_method: The method to use to compute the log determinant of the
            covariance matrix. One of "svd", "cholesky", "torch". Defaults to "svd".
                - svd: Use the singular value decomposition to compute the log determinant.
                - cholesky: Use the Cholesky decomposition to compute the log determinant.
                - torch: Use the torch.logdet function to compute the log determinant.
        augment_method: The method to use to augment the model with the new points
            and computing the posterior covariance.
            One of "naive", "cholesky", "get_fantasy_model". Defaults to "naive".
                - naive: Adds the new points to the training data and recomputes the
                    posterior from scratch.
                - cholesky: Uses a low rank update to the Cholesky decomposition of
                    the train-train covariance matrix to compute the posterior covariance.
                - get_fantasy_model: Uses the get_fantasy_model method of GPyTorch
                    to compute the posterior.
    """
    def __init__(
        self,
        model: Model,
        temperature: Union[float, np.ndarray, torch.Tensor],
        kernel_amplitude: float = 1.0,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        logdet_method: Union[str, LogDetMethod] = "svd",
        augment_method: Union[str, AugmentedPosteriorMethod] = "naive",
        **kwargs,
    ) -> None:

        super().__init__(model=model, posterior_transform=posterior_transform)

        self.logdet_method = LogDetMethod[logdet_method.upper()] if isinstance(logdet_method, str) else logdet_method
        self.augment_method = AugmentedPosteriorMethod[augment_method.upper()] if isinstance(augment_method, str) else augment_method
 
        self.kernel_amplitude = kernel_amplitude
 
        temperature = temperature * (self.kernel_amplitude)**(1/2)
        if type(temperature) == float and temperature == 0:
            temperature = 1e-10
        elif type(temperature) in [np.ndarray, torch.Tensor] and any(temperature == 0):
            temperature[temperature == 0] = 1e-10

        # define beta=1/T, push to device and register as buffer
        self.register_buffer("beta", torch.as_tensor(1/temperature).to( model.covar_module.raw_outputscale.device))

        if isinstance(temperature, np.ndarray) or isinstance(temperature, torch.Tensor):
            self.individual_beta = True #beta is an array of length q
            # unless the array contains exactly one element
            if len(temperature) == 1:
                self.individual_beta = False 
        else:
            self.individual_beta = False # beta is a scalar
        
        self.maximize = maximize


        if self.augment_method == AugmentedPosteriorMethod.CHOLESKY:
            self.predictor = GPPosteriorPredictor(
                model.covar_module,
                model.mean_module,
                model.likelihood.noise_covar,
                train_X=model.train_inputs[0],
                train_y=model.train_targets,
            )
        elif self.augment_method == AugmentedPosteriorMethod.NAIVE:
            # for augmentation, we keep a copy of the original model
            # if we make a copy in the forward pass only, we get a memory leak
            self.augmented_model = deepcopy(model)

        self.summary_fn = torch.sum

        

    
    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the free energy of the candidate set X.
        Args:
            X: A `(b1 x ... bk) x q x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of BOSS values at the
            given design points `X`.
        """

        # required for gradients in augmented GPs.
        with settings.detach_test_caches(False):
            self.model.eval()

            
            # Entropy term.
            f_preds = self.model(X) # this gets p(f* | x*, X, y) with x* being test points.
            posterior_cov = f_preds.covariance_matrix #  == C'_D
            posterior_means = f_preds.mean

            if self.augment_method == AugmentedPosteriorMethod.NAIVE:
                # augment the training data with the test points
                X_train_original = self.model.train_inputs[0]
                Y_train_original = self.model.train_targets
                X_train = X_train_original.expand(X.shape[0], X_train_original.shape[0], X_train_original.shape[1]) # (n_batch, n_train, dim)
                Y_train = Y_train_original.expand(X.shape[0], Y_train_original.shape[0]) # (n_batch, n_train)
                X_train_augmented = torch.cat([X_train, X], dim=1) # (n_batch, n_train + n_aug, dim)
                Y_train_augmented = torch.cat([Y_train, torch.zeros_like(X[:,:,1])], dim=1) # (n_batch, n_train + n_aug)
                self.augmented_model.set_train_data(X_train_augmented, Y_train_augmented, strict=False)

                f_preds_augmented = self.augmented_model(X)
                posterior_means_augmented = f_preds_augmented.mean
                posterior_cov_augmented = f_preds_augmented.covariance_matrix

            elif self.augment_method == AugmentedPosteriorMethod.CHOLESKY:
                posterior_cov_augmented = self.predictor.augmented_covariance(X)
                posterior_means_augmented = torch.zeros_like(posterior_means) #not used


            elif self.augment_method == AugmentedPosteriorMethod.GET_FANTASY_MODEL:
                fantasy_model = self.model.get_fantasy_model(X, torch.zeros_like(X[:,:,1]))
                f_preds_augmented = fantasy_model(X)
                posterior_means_augmented = f_preds_augmented.mean
                posterior_cov_augmented = f_preds_augmented.covariance_matrix




            if self.logdet_method == LogDetMethod.CHOLESKY:
                # use cholesky decomposition to compute logdet, fallback to svd if fails
                with settings.cholesky_max_tries(1):
                    try:
                        posterior_cov_logdet = f_preds.lazy_covariance_matrix.logdet()
                        # also trigger exception when any nan in result
                        if torch.isnan(posterior_cov_logdet).any():
                            raise Exception('nan in logdet')
                        elif torch.isinf(posterior_cov_logdet).any():
                            raise Exception('inf in logdet')
                    except Exception as e:
                        print(f'Cholesky failed: {e}')
                        _, s, _ = torch.svd(posterior_cov)
                        posterior_cov_logdet = torch.sum(torch.log(s), dim=-1)

                    try:
                        if self.augment_method == AugmentedPosteriorMethod.CHOLESKY:
                            # there is no lazy_covariance_matrix if we use cholesky augmentation
                            chol = psd_safe_cholesky(posterior_cov_augmented)
                            posterior_cov_augmented_logdet = chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) * 2
                            # posterior_cov_augmented_logdet = torch.logdet(posterior_cov_augmented)
                        else:
                            posterior_cov_augmented_logdet = f_preds_augmented.lazy_covariance_matrix.logdet()
                        # also trigger exception when any nan in result
                        if torch.isnan(posterior_cov_augmented_logdet).any():
                            raise Exception('nan in logdet')
                        elif torch.isinf(posterior_cov_augmented_logdet).any():
                            raise Exception('inf in logdet')
                    except Exception as e:
                        print(f'Cholesky failed: {e}')
                        _, s, _ = torch.svd(posterior_cov_augmented)
                        posterior_cov_augmented_logdet = torch.sum(torch.log(s), dim=-1)

            elif self.logdet_method == LogDetMethod.SVD:
                # use svd to compute logdet
                s = torch.linalg.svdvals(posterior_cov_augmented)
                # s[s==0] = 1e-20 # avoid nan # same but autograd friendly
                s = torch.where(s==0, torch.ones_like(s) * 1e-20, s)
                posterior_cov_augmented_logdet = torch.sum(torch.log(s), dim=-1)
                s = torch.linalg.svdvals(posterior_cov)
                s = torch.where(s==0, torch.ones_like(s) * 1e-20, s) # avoid nan
                posterior_cov_logdet = torch.sum(torch.log(s), dim=-1)

            elif self.logdet_method == LogDetMethod.TORCH:
                posterior_cov = posterior_cov + torch.eye(posterior_cov.shape[-1], device=posterior_cov.device) * 1e-06
                posterior_cov_logdet =  torch.logdet(posterior_cov)    
                posterior_cov_augmented = posterior_cov_augmented + torch.eye(posterior_cov_augmented.shape[-1], device=posterior_cov_augmented.device) * 1e-0
                posterior_cov_augmented_logdet = torch.logdet(posterior_cov_augmented)
            else:
                raise NotImplementedError(f'logdet method {self.logdet_method} not implemented')

            if torch.isinf(posterior_cov_augmented_logdet).any():
                print('augmented cov logdet is inf')
            if torch.isinf(posterior_cov_logdet).any():
                print('cov logdet is inf')


            if torch.isnan(posterior_cov_augmented_logdet).any():
                print('augmented cov logdet is nan')
            if torch.isnan(posterior_cov_logdet).any():
                print('cov logdet is nan')

            information_gain = 0.5* (posterior_cov_logdet - posterior_cov_augmented_logdet)

            
            if self.individual_beta:
                # multiply each of the posterior means with its own beta
                # posterior_means is (num_restarts, q) --> broadcast over num_restarts
                posterior_means = posterior_means * self.beta

                summary_posterior = self.summary_fn(posterior_means, dim=1)
                summary_augmented = self.summary_fn(posterior_means_augmented, dim=1) 

                if self.maximize:
                    # maximize fn value + gain
                    acq_value =  summary_posterior + information_gain
                else:
                    acq_value =  (-1) * summary_posterior + information_gain

                acq_value += summary_augmented*0 # this prevents memory leaks.

            else:

                summary_posterior = self.summary_fn(posterior_means, dim=1)
                summary_augmented = self.summary_fn(posterior_means_augmented, dim=1) 

            
                if self.maximize:
                    # maximize fn value + gain
                    acq_value =  self.beta * summary_posterior + information_gain
                else:
                    acq_value =  (-1) * self.beta * summary_posterior + information_gain

                acq_value += summary_augmented*0 # this prevents memory leaks.


            return acq_value


    def _get_augmented_covariance_logdet_lowrank_update(self, X: torch.Tensor):

        # access the train_train_covar cache from self.model

        # TODO check whether we are working with LOVE or not.
        pass


    # NOTE the methods below are only there for better readability.
    # Due to memory leaks and for numerical reasons, they are not used in the actual code.
    # We keep them here for future reference, as a minimal example of how to compute the
    # two terms of the BEE-BOSS acquisition function.
    @t_batch_mode_transform()
    def compute_energy(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the energy of the candidate set X. 
        Args:
            X: A `(b1 x ... bk) x q x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of BOSS values at the
            given design points `X`.
        """
        with settings.detach_test_caches(False):
            self.model.eval()

            # Enthalpy term.
            f_preds = self.model(X) # this gets p(f* | x*, X, y) with x* being test points.
            posterior_means = f_preds.mean

            summary = self.summary_fn(posterior_means, dim=1) 


            if not self.maximize:
                # minimize fn value + maximize gain
                summary = (-1) * summary
            
            return summary
        

    @t_batch_mode_transform()
    def compute_entropy(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the energy of the candidate set X. 
        Args:
            X: A `(b1 x ... bk) x q x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of information gain values at the
            given design points `X`.
        """
        with settings.detach_test_caches(False):
            self.model.eval()

            # Entropy term.
            f_preds = self.model(X) # this gets p(f* | x*, X, y) with x* being test points.
            posterior_cov = f_preds.covariance_matrix #  == C'_D

            ## augment observations with x' and dummy y' (because gpytorch requires them)
            model_augmented = self.model.get_fantasy_model(X, torch.zeros_like(X[:,:,1])) 
            f_preds = model_augmented(X)
            posterior_cov_augmented = f_preds.covariance_matrix # == C'_D_D'
            posterior_cov_augmented = posterior_cov_augmented + torch.eye(posterior_cov_augmented.shape[-1], device=posterior_cov_augmented.device) * 1e-04

            information_gain = torch.logdet(posterior_cov) - torch.logdet(posterior_cov_augmented)


            return information_gain






