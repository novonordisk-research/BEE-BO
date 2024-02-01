from typing import List, Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction


class EmbeddedHartmann(SyntheticTestFunction):
    """
    The embedded Hartmann test function.
    
    We take the Hartmann function, and concatenate d-d_hartmann dummy dimensions to it.
    """

#  Most commonly used is the six-dimensional version (typically evaluated on
#     `[0, 1]^6`):

#         H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

#     H has a 6 local minima and a global minimum at

#         z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

#     with `H(z) = -3.32237`.
#     """

    def __init__(
        self,
        dim=100,
        dim_hartmann=6,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim_hartmann not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {dim_hartmann} not defined")
        self.dim = dim
        self.dim_hartmann = dim_hartmann
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.dim)]
        # optimizers and optimal values for dim=4 not implemented
        optvals = {3: -3.86278, 6: -3.32237}
        optimizers = {
            3: [(0.114614, 0.555649, 0.852547) + (0.0,) * (dim - 3)],
            6: [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) + (0.0,) * (dim - 6)],
        }
        self._optimal_value = optvals.get(self.dim_hartmann)
        self._optimizers = optimizers.get(self.dim_hartmann)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim_hartmann == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim_hartmann == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim_hartmann == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))

    @property
    def optimal_value(self) -> float:
        if self.dim_hartmann == 4:
            raise NotImplementedError()
        return super().optimal_value

    @property
    def optimizers(self) -> torch.Tensor:
        if self.dim_hartmann == 4:
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:

        # cut dummy dimensions
        X = X[..., : self.dim_hartmann]

        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        return H