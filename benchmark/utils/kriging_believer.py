"""
The kriging believer heuristic strategy.
"""
import gpytorch
from botorch.generation import gen_candidates_torch
from copy import deepcopy
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import normalize
import torch

def get_candidates_kriging_believer(model, config, bounds, initial_conditions=None):
    """Generate candidates using the acquisition function.

    Parameters
    ----------
        model: 
            A GP model.
        config: dict
            Configuration dictionary.
        bounds: torch.Tensor 
            The bounds of the search space.
        initial_conditions: torch.Tensor 
            The initial conditions. Defaults to None. If this is provided,
            the raw_samples argument of optimize_acqf is ignored.

    Returns
    -------
        points: torch.Tensor
            The generated candidates.
    """
    model = deepcopy(model)
    batch = []

    for q in range(config['batch_size']):
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.acquisition import LogExpectedImprovement
        best_f = model.train_targets.max().item()
        acq = LogExpectedImprovement(model, best_f)

        # this is necessary for the root decomposition in the MC sampler to work as expected
        with gpytorch.settings.max_root_decomposition_size(max(100, config['batch_size'])):#, gpytorch.settings.fast_pred_var(True):

            generator = None
            gen_candidates = gen_candidates_torch if config['opt']=='torch' else None
            point, value = optimize_acqf(
                acq, 
                q=1, 
                num_restarts=config['num_restarts'], 
                bounds=normalize(bounds, bounds), 
                raw_samples=config['raw_samples'] if initial_conditions is None else None, 
                gen_candidates= gen_candidates,
                batch_initial_conditions=initial_conditions,
                generator = generator,
                )
            
            pred = model(point).mean
            batch.append(point.detach().cpu())
            model = model.get_fantasy_model(point, pred)

        
    return torch.cat(batch, dim=0), value