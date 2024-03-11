"""
Do cholesky-based GP predictions, allow for low-rank updates.

GPytorch does not support low-rank updates with Cholesky as far as I can tell.
"""
import torch

# TODO make sure batch mode works correctly for all methods.
# Add checks and enforce that we use batch mode for all inputs to forward,
# plus shape checks for stuff provided in init. probably don't want batch mode there.

class GPPosteriorCovar():

    def __init__(
            self,
            kernel,
            noise_module,
            train_X,
            train_train_covar,
            train_train_covar_chol = None
            ) -> None:
        
        self.kernel = kernel
        self.noise_module = noise_module
        self.train_X = train_X

        # NOTE train_train_covar has sigma^2 added to the diagonal.
        self.train_train_covar = train_train_covar
        if train_train_covar_chol is None:
            self.train_train_covar_chol = torch.linalg.cholesky(self.train_train_covar)
        else:
            self.train_train_covar_chol = train_train_covar_chol

    # i see no reason why we should not compile this.
    # the shape of X won't change in repeated calls when
    # optimizing X.
    # @torch.compile
    def forward(self, X):
        """
        Basic code taken from exact_predictive_covar in GPytorch.
        """

        test_test_covar = self.kernel(X).to_dense()
        test_train_covar = self.kernel(X, self.train_X).to_dense()
        train_test_covar = test_train_covar.transpose(-1, -2)

        covar_correction_rhs = torch.cholesky_solve(train_test_covar, self.train_train_covar_chol)

        posterior = test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1)

        return posterior

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def augmented_covariance(self, new_X):
        """
        Add new_X to the training set, then compute posterior covariance of new_X.
        Use low-rank update to avoid recomputing the entire cholesky decomposition.
        """
        # TODO maybe use some indexing to avoid too many kernel calls.
        test_test_covar = self.kernel(new_X).to_dense()
        test_train_covar = self.kernel(new_X, torch.cat([self.train_X, new_X], axis=0)).to_dense()
        train_test_covar = test_train_covar.transpose(-1, -2)

        train_train_covar_chol_aug = self.update_chol(
            self.train_train_covar_chol,
            self.kernel(self.train_X, new_X).to_dense(),
            test_test_covar + self.noise_module(new_X).to_dense()
        )

        covar_correction_rhs = torch.cholesky_solve(train_test_covar, train_train_covar_chol_aug)

        posterior = test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1)

        return posterior

    @staticmethod
    def update_chol(L, B, C):
        """Update cholesky decomposition of M to M_aug.

        Args:
            L (np.ndarray): Cholesky decomposition of M (n, n)
            B (np.ndarray): old-new covar (n, q)
            C (np.ndarray): new-new covar (q, q) 
            NOTE: C needs to include the noise on the diagonal.

        Returns:
            L_aug: Cholesky decomposition of M_aug (n+q, n+q)
        """
        # print(L.shape, B.shape, C.shape)
        X = torch.linalg.solve_triangular(L, B, upper=False).T

        # Calculate S (Schur complement)
        S = C - X @ X.T

        # Calculate Y
        Y = torch.linalg.cholesky(S, upper=False)


        # Combine as [[L, X], [0, Y]]
        L_aug = torch.cat([
            torch.cat([L, torch.zeros_like(B)], axis=1), 
            torch.cat([X, Y], axis=1)
            ], axis=0)
        
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

    # compute vectors.
    # TODO shape check - maybe need some transposes/dummy dimensions.
    delta_k_aa = kernel(new_x) - kernel(x_augmented[new_x_idx])

    delta_k = kernel(x_train, new_x) - kernel(x_train, x_augmented[new_x_idx]) - 0.5 * delta_k_aa * kronecker_delta
    delta_kA = kernel(x_aug_replaced, new_x) - kernel(x_augmented, x_augmented[new_x_idx]) - 0.5 * delta_k_aa * kronecker_delta_augmented

    delta_mA =
    pass