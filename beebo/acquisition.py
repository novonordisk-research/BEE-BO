import warnings
from copy import deepcopy
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import BotorchWarning
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from linear_operator.utils.cholesky import psd_safe_cholesky

from .utils.cholesky_inference import GPPosteriorPredictor


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
    # TODO replace with a wrapper for GET_FANTASY_STRATEGY --> skip the internal model deepcopy

from linear_operator.operators import (DiagLinearOperator,
                                       LowRankRootLinearOperator)
from linear_operator.utils.cholesky import psd_safe_cholesky


def stable_softmax(x: torch.Tensor, beta: float, f_max: float = None, eps=1e-6, alpha=0.05):
    
    if f_max is None:
        x_scaled = beta * x
        z = x_scaled - x_scaled.max(dim=-1, keepdim=True).values # (n x q+1)
        z_exp = z.exp()
        w = z_exp / z_exp.sum(dim=-1, keepdim=True) # (n x q)

        # ref_denominator = z_exp.sum(dim=-1, keepdim=True)
        # ref_weights = z_exp / z_exp.sum(dim=-1, keepdim=True)
    else:
        x_scaled = beta * x
        beta_delta_x = x_scaled - x_scaled.max(dim=-1, keepdim=True).values
        beta_delta_fmax = beta * f_max - x_scaled.max(dim=-1, keepdim=True).values
        # import ipdb; ipdb.set_trace()
        denominator = beta_delta_x.exp().sum(dim=-1, keepdim=True)

        # get min of
        # 1/(1-alpha))* denominator and
        # beta_delta_fmax.exp()
        g = torch.stack([denominator * (1-alpha)/alpha, beta_delta_fmax.exp()], dim=-1)
        g = g.min(dim=-1).values

        w = beta_delta_x.exp() / (denominator+g)
        # w_bad = beta_delta_x.exp() / (denominator+beta_delta_fmax.exp())

        # d_tilde = 1./(1+ (beta_delta_fmax - denominator.log()).exp()) 
        # d_tilde = d_tilde + eps

        # w = d_tilde * (beta_delta_x - denominator.log()).exp()

    return w

def softmax_expectation_a_is_mean(mvn, softmax_beta, f_max=None):

    means = mvn.mean # (n x q)
    covar = mvn.covariance_matrix # (n x q x q)
    lazy_covar = mvn.lazy_covariance_matrix # (n x q x q)

    w = stable_softmax(means, softmax_beta, f_max)

    # threshold = torch.tensor(softmax_beta*f_max).exp() if f_max is not None else 0
    # # evaluate softmax at expansion point
    # exps = (softmax_beta*means).exp()
    # w = exps / (exps.sum(dim=1) + threshold).unsqueeze(-1) # (n x q)

    W = DiagLinearOperator(w) - LowRankRootLinearOperator(w.unsqueeze(-1)) # (n x q x q)

    U_inv = DiagLinearOperator(torch.ones(covar.shape[1], device=means.device)) + softmax_beta**2 * lazy_covar @ W
    # C_update = U_inv.solve(covar) # we avoud doing a solve for C_update - we keep this factorized and do U_inv.solve(x) whereever it appears

    # nu = beta * C_update @ (e-w) + means
    # nu = beta * (C_update @ e - C_update @ w) + means
    # C_update @ e = select C_update[:,i] for each i
    # from each col, subtract C_update @ w, then multiply by beta, then add means

    # nu_i_matrix = softmax_beta * (C_update - C_update @ w.unsqueeze(-1)) + means.unsqueeze(-1)
    # ^checked, matches previous implementation. But we can do numerical tricks to avoid a solve for C_update.
    # avoid doing a solve for C_update.
    col_difference = U_inv.solve(covar - covar @ w.unsqueeze(-1)) # this is C_update - C_update @ w.unsqueeze(-1)
    nu_i_matrix = softmax_beta *  col_difference + means.unsqueeze(-1) # (n x q x q)
    

    # testright = C_update[0,:,[1]] - (C_update @ w.unsqueeze(-1))[0] #(q,1)
    # testright_2 = (C_update[[0]] - C_update[[0]] @ w[[0]].unsqueeze(-1))[:,:,1] #(q)
    # testright_3 = (C_update - C_update @ w.unsqueeze(-1))[0,:,1] #(q)

    # i = 4
    # base = torch.zeros(10, device=means.device)
    # base[i] = 1
    # ctest = 0.5*softmax_beta**2 *(base-w[0]).unsqueeze(-1).T @ C_update[0] @ (base-w[0]).unsqueeze(-1) # (q,1)
    # c_i_vector[0,i] must equal ctest. good.

    # col_difference = C_update - C_update @ w.unsqueeze(-1) # (n x q x q)
    # ^again, don't want for numerical reasons.
    c_i_vector = 0.5*softmax_beta**2 * (
        torch.diagonal(col_difference, dim1=-2, dim2=-1)
        - (w.unsqueeze(-1).mT @ col_difference).squeeze() # (n x 1 x q)
    ) # n x q
    # U_inv.to_dense().logdet() is much more stable than U_inv.logdet() for some reason
    # K = (1/U_inv.logdet().exp()).sqrt()
    K = (1/U_inv.to_dense().det()).sqrt()
    expectation = K * ((w.log() + c_i_vector).exp() * torch.diagonal(nu_i_matrix, dim1=-2, dim2=-1)).sum(dim=-1)
    return expectation



    # this is the "old" math - stuff above is Jesper 18 Apr rewrite
    shortcut_nu_prime = means.unsqueeze(-1) - softmax_beta * (C_update @ w.unsqueeze(-1))
    # beta*C_update * e(i) + nu_prime --> ith column of c_update + nu_prime

    # index as [b,:,i] to get nu_i for each i
    shortcut_nu_i_matrix = softmax_beta * C_update + shortcut_nu_prime # (n x q x q)

    shortcut_c_prime  = (softmax_beta * w.unsqueeze(-1).mT @ means.unsqueeze(-1) + # (n x 1 x 1)
              0.5 * softmax_beta**2 * w.unsqueeze(-1).mT @ U_inv @ lazy_covar.solve( w.unsqueeze(-1))
    )

    C_update_sigma = torch.diagonal(C_update, dim1=-2, dim2=-1)
    shortcut_c_i_vector = (
        -softmax_beta * means + 0.5 * softmax_beta**2 * C_update_sigma # this part batches correctly, n x q
        + (softmax_beta * shortcut_nu_prime + shortcut_c_prime).squeeze(-1)
    ) 
    K = (1/U_inv.logdet().exp()).sqrt()
    shortcut_expectation = K * ((w.log() + shortcut_c_i_vector).exp() * torch.diagonal(shortcut_nu_i_matrix, dim1=-2, dim2=-1)).sum(dim=-1)
    return shortcut_expectation

def softmax_expectation(mvn: MultivariateNormal, a: torch.Tensor, softmax_beta: float, f_max: float = None):

    if False:

        means = mvn.mean # (n x q)
        covar = mvn.covariance_matrix # (n x q x q)
        lazy_covar = mvn.lazy_covariance_matrix # (n x q x q)

        threshold = torch.tensor(softmax_beta*f_max).exp() if f_max is not None else 0
        # evaluate softmax at expansion point
        exps = (softmax_beta*a).exp()
        w = exps / (exps.sum(dim=1) + threshold).unsqueeze(-1) # (n x q)
        w = stable_softmax(means, softmax_beta, f_max) # (n x q)
        # W_torch = torch.diag_embed(w) - w.unsqueeze(-1) @ w.unsqueeze(-2) #isclose OK.
        W = DiagLinearOperator(w) - LowRankRootLinearOperator(w.unsqueeze(-1)) # (n x q x q)

        U_inv = DiagLinearOperator(torch.ones(covar.shape[1], device=a.device)) + softmax_beta**2 * lazy_covar @ W
        # U_inv_torch = torch.eye(*covar.shape[1:]) + softmax_beta**2 * covar @ W_torch # isclose OK

        C_update = U_inv.solve(covar) #TODO this fails at large beta

        # U = torch.inverse(torch.eye(*covar.shape[1:]) + softmax_beta**2 * covar @ W) 
        # C_update_torch = U @ covar # isclose not OK at default tolerance, but at 1e-03 it is OK

        # updated means
        # NOTE nu_prime should be the mean if beta=0 and a=mean
        nu_prime = C_update @ (-softmax_beta * w.unsqueeze(-1) + lazy_covar.solve(means.unsqueeze(-1)) + softmax_beta**2 * W @ a.unsqueeze(-1))
        # nu_i is beta * C_update[:,i] + nu_prime
        # nu_i_matrix = C_update - nu_prime # n x q x q
        # index as [b,:,i] to get nu_i for each i
        nu_i_matrix = softmax_beta * C_update + nu_prime

        c_prime = (
            softmax_beta  * w.unsqueeze(-1).mT @ a.unsqueeze(-1) 
            +  0.5 * nu_prime.mT @ U_inv @ lazy_covar.solve(nu_prime) # THIS TERM goes wild
            - softmax_beta**2 * 0.5 * a.unsqueeze(-1).mT @ W @ a.unsqueeze(-1) 
            - 0.5 * means.unsqueeze(-1).mT @ lazy_covar.solve(means.unsqueeze(-1)) 

        ) # (n x 1 x 1)

        C_update_sigma = torch.diagonal(C_update, dim1=-2, dim2=-1) #(n x q)
        c_i_vector = (
            -softmax_beta * a + 0.5 * softmax_beta**2 * C_update_sigma
            + (softmax_beta * nu_prime + c_prime).squeeze(-1)
        ) # (n x q)
        # index as [b,i] to get c_i for each i

        K = (1/U_inv.logdet().exp()).sqrt()
        if torch.isnan(K).any():
            raise Exception('nan in K logdet')

        expectation = K * ((w.log() + c_i_vector).exp() * torch.diagonal(nu_i_matrix, dim1=-2, dim2=-1)).sum(dim=-1) # (n)


    #### Debug code - compare different ways of computing the expectation
    # a=mu shortcut code:
    shortcut_expectation = softmax_expectation_a_is_mean(mvn, softmax_beta, f_max)

    # TODO fix general case another time. c' explodes.
    if False:
        with torch.no_grad():
            all_samples = []
            for i in range(10000):
                sample = mvn.sample()
                # softmax sum of sample
                # exps = (softmax_beta*sample).exp()
                # w = exps / (exps.sum(dim=1) + threshold).unsqueeze(-1)
                w = stable_softmax(sample, softmax_beta, f_max)
                ssum = (w * sample).sum(dim=1)

                all_samples.append(ssum)

        approx = torch.stack(all_samples).mean(dim=0)

        print('sampled, shortcut, mean, max. beta:', softmax_beta, 'mean diff:', (approx - shortcut_expectation).abs().mean().item())
        # 4 column layout
        import numpy as np
        np.set_printoptions(suppress=True)
        print_tensor = torch.cat([approx.unsqueeze(-1), shortcut_expectation.unsqueeze(-1), a.mean(dim=-1).unsqueeze(-1), a.max(dim=-1).values.unsqueeze(-1)], dim=-1)
        print_array = np.array_str(print_tensor.detach().cpu().numpy(), precision=3, suppress_small=True)
        print(print_array)

    expectation = shortcut_expectation

    if torch.isnan(expectation).any():
        import ipdb; ipdb.set_trace()
        raise Exception('nan in expectation')
    
    if torch.isinf(expectation).any():
        raise Exception('inf in expectation')

    return expectation



class EnergyFunction(Enum):
    """Used to specify the energy function to be used in the BEE-BO acquisition function."""
    SOFTMAX = "softmax"
    SUM = "sum"


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
        energy_function: The energy function to use in the BEE-BO acquisition function.
            One of "softmax", "sum". Defaults to "sum".
                - softmax: Uses the softmax energy function.
                - sum: Uses the sum energy function.
        **kwargs: Additional arguments to be passed to the energy function.
    """
    def __init__(
        self,
        model: Model,
        temperature: float,
        kernel_amplitude: float = 1.0,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[torch.Tensor] = None,
        maximize: bool = True,
        logdet_method: Union[str, LogDetMethod] = "svd",
        augment_method: Union[str, AugmentedPosteriorMethod] = "naive",
        energy_function: Union[str, EnergyFunction] = "sum",
        **kwargs,
    ) -> None:

        super().__init__(model=model, posterior_transform=posterior_transform)

        self.logdet_method = LogDetMethod[logdet_method.upper()] if isinstance(logdet_method, str) else logdet_method
        self.augment_method = AugmentedPosteriorMethod[augment_method.upper()] if isinstance(augment_method, str) else augment_method
        self.energy_function = EnergyFunction[energy_function.upper()] if isinstance(energy_function, str) else energy_function

        if self.energy_function == EnergyFunction.SOFTMAX:
            self.softmax_beta = kwargs.get("softmax_beta", 1.0)
            self.f_max = kwargs.get("f_max", None)
 
        self.kernel_amplitude = kernel_amplitude
 
        self.temperature = temperature * (self.kernel_amplitude)**(1/2)
        
        self.maximize = maximize

        self.set_X_pending(X_pending)


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

        # self.summary_fn = torch.sum

        

    @concatenate_pending_points    
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

            

            if self.energy_function == EnergyFunction.SUM:
                summary_posterior = torch.sum(posterior_means, dim=1)
                summary_augmented = torch.sum(posterior_means_augmented, dim=1)
            elif self.energy_function == EnergyFunction.SOFTMAX:
                summary_posterior = softmax_expectation(f_preds, a=f_preds.mean, softmax_beta=self.softmax_beta, f_max=self.f_max)
                summary_posterior = summary_posterior * f_preds.mean.shape[1] # multiply by q to make it scale linearly with q, like logdet

                # this is a dummy thing for memory leaks
                summary_augmented = torch.sum(posterior_means_augmented, dim=1)

        
            if self.maximize:
                # maximize fn value + gain
                acq_value =  summary_posterior + self.temperature * information_gain
            else:
                acq_value =  (-1) * summary_posterior + self.temperature * information_gain

            acq_value += summary_augmented*0 # this prevents memory leaks.

            # print('acq', 'info gain', 'expect.', 'sum','max')
            # print_array = torch.stack([acq_value, information_gain, summary_posterior, posterior_means.sum(1), posterior_means.max(1).values], dim=1) # (num_restarts, 5)
            # print_array = np.array_str(print_array.detach().cpu().numpy().mean(axis=0), precision=3, suppress_small=True)
            # print(print_array)


            return acq_value

    # NOTE the base AnalyticAcquisitionFunction class does not support X_pending
    def set_X_pending(self, X_pending: Optional[torch.Tensor] = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:
            # when doing sequential, stuff will have gradients. no point in
            # warning about it.
            # if X_pending.requires_grad:
            #     warnings.warn(
            #         "Pending points require a gradient but the acquisition function"
            #         " will not provide a gradient to these points.",
            #         BotorchWarning,
            #     )
            self.X_pending = X_pending.detach().clone()
        else:
            self.X_pending = X_pending


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






