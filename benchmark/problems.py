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
    

class RotatedEmbeddedHartmann(SyntheticTestFunction):
    """
    https://arxiv.org/pdf/2103.00349.pdf "D.4 Rotated Hartmann"
    
    We take the Hartmann function and add a projection on top.
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
        seed = 123,
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
        self.projection_matrix = torch.randn(dim, dim_hartmann, generator=torch.Generator().manual_seed(seed)) // dim_hartmann**0.5
        self.bias = torch.rand(dim, generator=torch.Generator().manual_seed(seed))
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

        # project
        X = X @ self.projection_matrix - self.bias

        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        return H

# https://github.com/zi-w/Ensemble-Bayesian-Optimization/blob/master/test_functions/push_function.py
try:
    from utils.push_utils import b2WorldInterface, make_base, create_body, end_effector, run_simulation
    import concurrent.futures
except ImportError:
    pass
class RobotPushing(SyntheticTestFunction):
    def __init__(
        self,
        dim=14,
        seed = 123,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        
        if dim != 14:
            raise ValueError(f"RobotPushing with dim {dim} not defined")
        
        self.dim = dim

        # domain of this function
        self.xmin = [-5., -5., -10., -10., 2., 0., -5., -5., -10., -10., 2., 0., -5., -5.]
        self.xmax = [5., 5., 10., 10., 30., 2.*np.pi, 5., 5., 10., 10., 30., 2.*np.pi, 5., 5.]

        # make bounds from xmin and xmax
        if bounds is None:
            bounds = [(self.xmin[i], self.xmax[i]) for i in range(self.dim)]
            

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]
        # self.fmax = np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) + np.linalg.norm(np.array(self.gxy2) - np.array(self.sxy2))
        self.fmax = 0

        self._optimal_value = self.fmax
        # self._optimizers = [self.gxy + self.gxy2]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    @property
    def optimal_value(self) -> float:
        return super().optimal_value

    @property
    def optimizers(self) -> torch.Tensor:
        return None
    #     return super().optimizers


    def get_push_reward(self,param_vector):
        rx = param_vector[0].item()
        ry = param_vector[1].item()
        xvel = param_vector[2].item()
        yvel = param_vector[3].item()
        simu_steps = int(float(param_vector[4].item()) * 10)
        init_angle = param_vector[5].item()
        rx2 = param_vector[6].item()
        ry2 = param_vector[7].item()
        xvel2 = param_vector[8].item()
        yvel2 = param_vector[9].item()
        simu_steps2 = int(float(param_vector[10].item()) * 10)
        init_angle2 = param_vector[11].item()
        rtor = param_vector[12].item()
        rtor2 = param_vector[13].item()

        initial_dist = self.fmax

        world = b2WorldInterface(True)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = \
            'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

        base = make_base(500, 500, world)
        body = create_body(base, world, 'rectangle', (0.5, 0.5), ofriction, odensity, self.sxy)
        body2 = create_body(base, world, 'circle', 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(world, (rx2,ry2), base, init_angle2, hand_shape, hand_size)
        (ret1, ret2) = run_simulation(world, body, body2, robot, robot2, xvel, yvel, \
                                      xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2)

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1) # distance to goal of object 1
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2) # distance to goal of object 2

        return ret1 + ret2 # total distance to goal: lower is better


    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        
        out = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        # for i in range(X.shape[0]):
        #     out[i] = self.get_push_reward(X[i])
        #     # print(out[i])
        jobs = []
        with concurrent.futures.ProcessPoolExecutor(16) as executor:
            for i in range(X.shape[0]):
                jobs.append(executor.submit(self.get_push_reward, X[i].detach()))

        for i, j in enumerate(jobs):
            out[i] = j.result()

        return out
    

from utils.rover_utils import create_large_domain
class Rover(SyntheticTestFunction):

    def __init__(
        self,
        dim: int = 60,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:

        # domain of this function
        self.s_range = np.array([[-0.1, -0.1], [1.1, 1.1]])
        self.dim = dim
        self.n_points = dim //2

        if bounds is None:
            bounds = [(self.s_range[0][0], self.s_range[1][0]) for i in range(self.n_points)] +  [(self.s_range[0][1], self.s_range[1][1]) for i in range(self.n_points)] 

        self.x_range = raw_x_range = np.repeat(self.s_range, self.n_points, axis=1)
        

        self._optimal_value = 5.0
        # self._optimizers = [torch.tensor([0., 0.])]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    @property
    def optimal_value(self) -> float:
        return super().optimal_value

    @property
    def optimizers(self) -> torch.Tensor:
        return None
    #     return super().optimizers



    def get_rover_cost(self, x):
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)

        domain = create_large_domain(force_start=False,
                                    force_goal=False,
                                    start_miss_cost=l2cost,
                                    goal_miss_cost=l2cost,
                                    n_points=self.n_points
                                    )
        x_norm = x * (self.x_range[1] - self.x_range[0]) + self.x_range[0]


        out = domain(x_norm) #+ f_max
        return out
        
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        
        out = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        jobs = []
        with concurrent.futures.ProcessPoolExecutor(16) as executor:
            for i in range(X.shape[0]):
                jobs.append(executor.submit(self.get_rover_cost, X[i].detach()))

        for i, j in enumerate(jobs):
            out[i] = j.result()

        return out




# def main():
#     def l2cost(x, point):
#         return 10 * np.linalg.norm(x - point, 1)

#     domain = create_large_domain(force_start=False,
#                                  force_goal=False,
#                                  start_miss_cost=l2cost,
#                                  goal_miss_cost=l2cost)
#     n_points = domain.traj.npoints

#     raw_x_range = np.repeat(domain.s_range, n_points, axis=1)

#     from ebo_core.helper import ConstantOffsetFn, NormalizedInputFn

#     # maximum value of f
#     f_max = 5.0
#     f = ConstantOffsetFn(domain, f_max)
#     f = NormalizedInputFn(f, raw_x_range)
#     x_range = f.get_range()

#     x = np.random.uniform(x_range[0], x_range[1])
#     print('Input = {}'.format(x))
#     print('Output = {}'.format(f(x)))
