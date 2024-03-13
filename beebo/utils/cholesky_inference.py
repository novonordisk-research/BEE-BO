"""
Do cholesky-based GP predictions, allow for low-rank updates.

GPytorch does not support low-rank updates with Cholesky as far as I can tell.
"""
import torch

# TODO heteroskedastic GP - make sure we add noise, and noise is called correctly.
# TODO can't handle multi task GP yet - catch exception on train_X

class GPPosteriorPredictor():
    """
    A convenience class for computing posterior covariances of a GP.
    This avoids using GPytorch's default forward pass so that we can
    do cholesky-based predictions and low rank updates.
    """

    def __init__(
            self,
            covar_module,
            mean_module,
            noise_module,
            train_X,
            train_y,
            ) -> None:
        
        self.covar_module = covar_module
        self.mean_module = mean_module
        self.noise_module = noise_module
        self.train_X = train_X
        self.train_y = train_y

        # prepare the cache for the posterior covariance - M^{-1}
        # We don't actually invert M, but we use the cholesky decomposition.
        noise = noise_module(train_X) # train_train_covar has sigma^2 added to the diagonal.
        self.train_train_covar = self.covar_module(train_X).to_dense() + noise.to_dense()
        self.train_train_covar_chol = torch.linalg.cholesky(self.train_train_covar)

        # prepare the cache for the posterior mean - M^{-1} * y
        train_mean = mean_module(train_X).squeeze(-1)
        train_labels_offset = (self.train_y.squeeze(-1) - train_mean).unsqueeze(-1)
        self.mean_cache = torch.cholesky_solve(train_labels_offset, self.train_train_covar_chol).squeeze(-1)


    # i see no reason why we should not compile this.
    # the shape of X won't change in repeated calls when
    # optimizing X.
    # @torch.compile #need to downgrade to python3.10
    def predict_covar(self, X, test_train_covar=None):
        """
        Basic code taken from exact_predictive_covar in GPytorch.

        NOTE this supports both batch mode (b,q,d) and single mode (q,d).
        """

        test_test_covar = self.covar_module(X).to_dense()

        if test_train_covar is None:
            test_train_covar = self.covar_module(X, self.train_X).to_dense()
        train_test_covar = test_train_covar.transpose(-1, -2)

        covar_correction_rhs = torch.cholesky_solve(train_test_covar, self.train_train_covar_chol)

        posterior = test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1)

        return posterior
    

    def forward(self, X):
        """
        Get mean and covariance of the GP at X.
        """
        test_train_covar = self.covar_module(X, self.train_X).to_dense()
        return self.predict_mean(X, test_train_covar), self.predict_covar(X, test_train_covar)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def predict_mean(self, X, test_train_covar=None):

        test_mean = self.mean_module(X)

        if test_train_covar is None:
            test_train_covar = self.covar_module(X, self.train_X).to_dense()

        res = (test_train_covar @ self.mean_cache).squeeze(-1)
        res = res + test_mean

        return res


    def augmented_covariance(self, new_X):
        """
        Add new_X to the training set, then compute posterior covariance of new_X.
        Use low rank update to avoid recomputing the entire cholesky decomposition.
        """
        # find the q-batch dimension.
        q_dim = 0 if len(new_X.shape) == 2 else 1
        
        if len(new_X.shape) > 2:
            # need batched train_X 
            train_X = self.train_X.unsqueeze(0).expand(new_X.shape[0], -1, -1)
        else:
            train_X = self.train_X

        # TODO maybe use some indexing to avoid too many kernel calls.
        test_test_covar = self.covar_module(new_X).to_dense()
        test_train_aug_covar = self.covar_module(new_X, torch.cat([train_X, new_X], axis=q_dim)).to_dense()
        train_aug_test_covar = test_train_aug_covar.transpose(-1, -2)

        train_train_covar_chol_aug = self.update_chol(
            self.train_train_covar_chol,
            self.covar_module(train_X, new_X).to_dense(), # old-new covar
            test_test_covar + self.noise_module(new_X).to_dense() # new-new covar with noise
        )

        covar_correction_rhs = torch.cholesky_solve(train_aug_test_covar, train_train_covar_chol_aug)
        posterior = test_test_covar + test_train_aug_covar @ covar_correction_rhs.mul(-1)

        return posterior

    @staticmethod
    def update_chol(L, B, C):
        """Update cholesky decomposition of M to M_aug.

        Args:
            L (np.ndarray): Cholesky decomposition of M (n, n) / (b, n, n)
            B (np.ndarray): old-new covar (n, q) / (b, n, q)
            C (np.ndarray): new-new covar (q, q) / (b, q, q)
            NOTE: C needs to include the noise on the diagonal.

        Returns:
            L_aug: Cholesky decomposition of M_aug (n+q, n+q) / (b, n+q, n+q)
        """
        if len(B.shape) > 2:
            # ensure B and C are both batch mode.
            assert B.shape[0] == C.shape[0]
            

        X = torch.linalg.solve_triangular(L, B, upper=False).transpose(-1,-2)

        # Calculate S (Schur complement)
        S = C - torch.matmul(X, X.transpose(-1,-2))#X @ X.T

        # Calculate Y
        Y = torch.linalg.cholesky(S, upper=False)

        # make L with a batch dim and repeat.
        if len(B.shape) > 2:
            L_broadcasted = L.unsqueeze(0)
            L_broadcasted = L_broadcasted.expand(B.shape[0], -1, -1)
        else:
            L_broadcasted = L


        # Combine as [[L, X], [0, Y]]
        L_aug = torch.cat([
            torch.cat([L_broadcasted, torch.zeros_like(B)], axis=-1), 
            torch.cat([X, Y], axis=-1)
            ], axis=-2)
        
        return L_aug
    

def update_covar_one_point(
        covar: torch.Tensor,
        x_train: torch.Tensor, # shape (N,d)
        x_augmented: torch.Tensor, # shape (N+Q,d)
        new_x: torch.Tensor, # shape (1,d)
        new_x_idx : int, # index of new_x in x_augmented
        kernel: torch.nn.Module,
):

    # e_a
    kronecker_delta = torch.zeros(x_augmented.shape[0] - x_train.shape[0]) # shape (Q,)
    kronecker_delta[new_x_idx - x_train.shape[0]] = 1

    kronecker_delta_augmented = torch.cat([torch.zeros(x_train.shape[0]), kronecker_delta], axis=0) # shape (N+Q)


    x_aug_replaced = x_augmented.clone()
    x_aug_replaced[new_x_idx] = new_x

    # compute delta vectors.
    # TODO shape check - maybe need some transposes/dummy dimensions.
    delta_k_aa = kernel(new_x) - kernel(x_augmented[new_x_idx])

    delta_k = kernel(x_train, new_x) - kernel(x_train, x_augmented[new_x_idx]) - 0.5 * delta_k_aa * kronecker_delta
    delta_k_A = kernel(x_aug_replaced, new_x) - kernel(x_augmented, x_augmented[new_x_idx]) - 0.5 * delta_k_aa * kronecker_delta_augmented

    delta_m_A = 0
    delta_k_D = 0

    # x being the batch points.
    # TODO clean up signature. need x_train, x, and new_x+idx
    delta_k_tilde = delta_k - kernel(x, x_train)
    covar_updated = covar + delta_k_tilde @ kronecker_delta.T + kronecker_delta @ delta_k_tilde.T

