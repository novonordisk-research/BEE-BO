'''
Perform a BO multi-round experiment on a test problem.
'''
import torch
import numpy as np
import json
import os
import argparse
import pandas as pd
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim.optimize import optimize_acqf
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_torch
from botorch import test_functions
from tqdm.auto import tqdm
import time
import random

from beebo import BatchedEnergyEntropyBO
from problems import EmbeddedHartmann
from utils.kriging_believer import get_candidates_kriging_believer
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_test_problem(config):

    if config['test_function'] == 'hartmann':
        test_fn = test_functions.Hartmann(dim=config['dim'], negate=True)
    elif config['test_function'] == 'styblinskitang':
        test_fn = test_functions.StyblinskiTang(dim=config['dim'], negate=True)
    elif config['test_function'] == 'rosenbrock':
        test_fn = test_functions.Rosenbrock(dim=config['dim'], negate=True)
    elif config['test_function'] == 'rastrigin':
        test_fn = test_functions.Rastrigin(dim=config['dim'], negate=True)
    elif config['test_function'] == 'ackley':
        test_fn = test_functions.Ackley(dim=config['dim'], negate=True)
    elif config['test_function'] == 'levy':
        test_fn = test_functions.Levy(dim=config['dim'], negate=True)
    elif config['test_function'] == 'shekel':
        test_fn = test_functions.Shekel(negate=True)
        if config['dim'] != 4:
            raise NotImplementedError('Shekel only implemented for dim=4')
    elif config['test_function'] == 'cosine':
        test_fn = test_functions.Cosine8() # this has 1 global maximum by default. No need to negate.
        if config['dim'] != 8:
            raise NotImplementedError('Cosine only implemented for dim=8')
    elif config['test_function'] == 'embeddedhartmann':
        test_fn = EmbeddedHartmann(dim=config['dim'], dim_hartmann=6, negate=True)
    elif config['test_function'] == 'powell':
        test_fn = test_functions.Powell(dim=config['dim'], negate=True)
    else:
        raise NotImplementedError(config['test_function'])

    return test_fn


def get_starting_points(n_points: int, bounds: torch.Tensor, seed: int = 123, optima: torch.Tensor = None, min_dist: float = 0.2):

    generator = torch.Generator().manual_seed(seed)

    dim = bounds.shape[1]
    point_counter = 0

    optima_unit_cube = normalize(optima, bounds) if optima is not None else None
    optima_unit_cube = optima_unit_cube.to(torch.get_default_dtype())

    all_train_x_raw = torch.zeros(n_points, dim).to(torch.get_default_dtype())
    while point_counter < n_points:
        train_x_raw = torch.rand(n_points, dim, generator=generator).to(torch.get_default_dtype())

        # reject points that don't have min_dist to optima (euclidean distance)
        if optima is not None:

            dist = torch.cdist(optima_unit_cube, train_x_raw)
            mask = dist.min(dim=0).values > min_dist
            train_x_raw = train_x_raw[mask]

            train_x_raw = train_x_raw[:n_points-point_counter]

        # unnormalize
        train_x_raw = unnormalize(train_x_raw, bounds)

        all_train_x_raw[point_counter:point_counter+train_x_raw.shape[0]] = train_x_raw

        point_counter += train_x_raw.shape[0]

    return all_train_x_raw



def get_acquisition_function(acq_fn, model, config, kernel_amplitude=None, bounds=None):
    if config['acq_fn'] == 'beebo':
        acq = BatchedEnergyEntropyBO(model, temperature=config['explore_parameter'], kernel_amplitude=kernel_amplitude, logdet_method=config['logdet_method'], custom_inference=config['custom_inference'])
    elif config['acq_fn'] == 'maxbeebo':
        acq = BatchedEnergyEntropyBO(
            model, 
            temperature=config['explore_parameter'], 
            kernel_amplitude=kernel_amplitude, 
            logdet_method=config['logdet_method'], 
            custom_inference=config['custom_inference'], 
            energy_function='softmax',
            f_max = model.train_targets.max().item(),
            softmax_beta = 1/(kernel_amplitude **(1/2))
        )
    elif config['acq_fn'] == 'qucb':
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.acquisition import qUpperConfidenceBound
        sampler = SobolQMCNormalSampler(1024)
        # NOTE we **2 the explore_param because we want to control the internal beta.sqrt() - 
        # see section in the paper on equivalence of BEE-BO temperature and UCB sqrt(beta)
        acq = qUpperConfidenceBound(model, config['explore_parameter']**2, sampler)
    elif config['acq_fn'] == 'qei':
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.acquisition import qLogExpectedImprovement
        sampler = SobolQMCNormalSampler(1024)
        best_f = model.train_targets.max().item()
        acq = qLogExpectedImprovement(model, best_f, sampler)
    elif config['acq_fn'] == 'gibbon':
        from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
        candidate_set = torch.rand(
            config['gibbon_n_candidates'], config['dim'], device=model.train_inputs[0].device, dtype=model.train_inputs[0].dtype)
        acq = qLowerBoundMaxValueEntropy(model, candidate_set)
    elif config['acq_fn'] == 'modifiedgibbon':
        from utils.modified_gibbon import qModifiedLowerBoundMaxValueEntropy
        candidate_set = torch.rand(
            config['gibbon_n_candidates'], config['dim'], device=model.train_inputs[0].device, dtype=model.train_inputs[0].dtype)
        acq = qModifiedLowerBoundMaxValueEntropy(model, candidate_set)
        acq.set_batch_size(config['batch_size'])
    else:
        raise NotImplementedError()

    return acq



def get_candidates(acq, config, bounds, initial_conditions=None):
    """Generate candidates using the acquisition function.

    Parameters
    ----------
        acq: botorch.acqusition.acquisition.AcquisitionFunction
            The acquisition function.
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

    # this is necessary for the root decomposition in the MC sampler to work as expected
    with gpytorch.settings.max_root_decomposition_size(max(100, config['batch_size'])):#, gpytorch.settings.fast_pred_var(True):

        generator = None
        gen_candidates = gen_candidates_torch if config['opt']=='torch' else None
        points, value = optimize_acqf(
            acq, 
            q=config['batch_size'], 
            num_restarts=config['num_restarts'], 
            bounds=normalize(bounds, bounds), 
            raw_samples=config['raw_samples'] if initial_conditions is None else None, 
            gen_candidates= gen_candidates,
            batch_initial_conditions=initial_conditions,
            generator = generator,
            sequential= True if config['acq_fn'] in  ['gibbon', 'modifiedgibbon'] else False,
            )
            

        return points.cpu(), value
    

def get_random_candidates(config, bounds):
    '''
    Generate random candidates.

    Parameters
    ----------
        config: dict
            Configuration dictionary.
        bounds: torch.Tensor
            The bounds of the search space.

    Returns
    -------
        points: torch.Tensor
            The generated candidates.
    '''
    dim = bounds.shape[1]
    points = torch.rand(config['batch_size'], dim)

    return points

def get_thompson_candidates(model, config, n_candidates: int = 10000):
    '''
    Generate candidates using Thompson sampling.

    Parameters
    ----------
        model: gpytorch.models.ExactGP
            The GP model.
        config: dict
            Configuration dictionary.
        n_candidates: int
            The number of sobol candidates to generate.

    Returns
    -------
        points: torch.Tensor
            The generated candidates.
    '''
    from botorch.generation import MaxPosteriorSampling
    from torch.quasirandom import SobolEngine

    X_cand = SobolEngine(config['dim']).draw(n_candidates)

    with gpytorch.settings.max_cholesky_size(float("inf")):
        with torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            X_next = thompson_sampling(X_cand.to(device), num_samples=config["batch_size"])

    return X_next.cpu()


def get_matern_kernel_with_gamma_prior(
    ard_num_dims: int, batch_shape = None
):
    r"""Constructs the Scale-Matern kernel that is used by default by
    several models. This uses a Gamma(3.0, 6.0) prior for the lengthscale
    and a Gamma(2.0, 0.15) prior for the output scale.

    Adapted from BoTorch source to use KeOps instead.
    """
    return gpytorch.kernels.ScaleKernel(
        base_kernel=gpytorch.kernels.keops.MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=gpytorch.priors.torch_priors.GammaPrior(3.0, 6.0),
        ),
        batch_shape=batch_shape,
        outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15),
    )


def run_one_round(test_problem, train_x: torch.Tensor, train_y: torch.Tensor, config, i):

    train_x = normalize(train_x, test_problem.bounds).to(device)
    train_y = standardize(train_y).to(device)


    # print('Setting up GP')
    if config['keops']:
        kernel = get_matern_kernel_with_gamma_prior(
                    ard_num_dims=train_x.shape[-1], # BoTorch default: 1 length scale per input dim.
                    batch_shape=train_x.shape[:-2],
                )
        model = SingleTaskGP(train_x.detach(),train_y.detach(), covar_module=kernel)
    else:
        model = SingleTaskGP(train_x.detach(),train_y.detach())
    gpytorch.settings.max_cholesky_size(2)


    if torch.get_default_dtype() == torch.float64:
        model = model.double()
        model.likelihood = model.likelihood.double()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.to(device)
    mll = mll.to(device)

    try:
        mll = fit_gpytorch_mll(mll)
    except RuntimeError as e:
        print('Fitting GP failed. Retrying. with torch')
        from botorch.optim.fit import fit_gpytorch_mll_torch
        mll = fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)

    # print(f'Fitted GP on data: {train_x.shape[0]} points, {train_x.shape[1]} dimensions. Memory usage {torch.cuda.memory_allocated()/1e9} GB')
    # save the model
    # saving as state_dict does not keep the train data, so we save the whole model.
    try:
        torch.save(model, os.path.join(config['out_dir'], 'model_round_'+str(i)+'.pt'))
    except AttributeError as e:
        print(e)
        # pykeops doesn't support pickling.
        torch.save(model.state_dict(), os.path.join(config['out_dir'], 'model_round_'+str(i)+'.pt'))
        # also save train data.
        torch.save([model.train_inputs, model.train_targets], os.path.join(config['out_dir'], 'model_round_'+str(i)+'_train_data.pt'))


    bounds = test_problem.bounds.to(device)
    

    if config['acq_fn'] == 'random':
        acq = None
        points = get_random_candidates(config, bounds)
    elif config['acq_fn'] == 'thompson':
        acq = None
        points = get_thompson_candidates(model, config, n_candidates=config['n_thompson_base_samples'])
    elif config['acq_fn'] == 'krigingbeliever':
        acq = None
        points, values = get_candidates_kriging_believer(model, config, bounds)
    else:
        acq = get_acquisition_function(config['acq_fn'], model, config, model.covar_module.outputscale.item(), bounds)

        points, value = get_candidates(acq, config, bounds)


    new_x_raw = unnormalize(points, test_problem.bounds)
    new_y_raw = test_problem(new_x_raw).unsqueeze(-1)
    # print(f'Got {len(new_x_raw)}  new points.')

    # prevent a memory leak here
    del acq
    model.to('cpu')


    return new_x_raw, new_y_raw, model


        


def run_bo_rounds(config):

    print('Using device:', device)

    
    test_problem = get_test_problem(config)

    if config['test_function'] == 'gaussian':
        config['test_problem_optima'] = test_problem.minima.detach().numpy().tolist()
        config['test_problem_amplitudes'] = test_problem.amplitudes.detach().numpy().tolist()

    # get start points
    train_x = get_starting_points(config['n_start_points'], test_problem.bounds, seed=config['seed'], optima=test_problem.optimizers, min_dist=config['init_min_distance'])
    train_y = test_problem(train_x).unsqueeze(-1)

    experiment_log = []
    
    os.makedirs(config['out_dir'], exist_ok=True)

    for i in range(train_x.shape[0]):
        # make a dict of each point and add to log
        point = {}
        point['round'] = 0
        point['y'] = train_y[i].item()
        for j in range(train_x.shape[1]):
            point['x'+str(j)] = train_x[i,j].item()
        point['time'] = time.time()
        experiment_log.append(point)
        
    
    start_time = time.time()
    for round in tqdm(range(config['n_rounds'])):

        
        new_x, new_y, model = run_one_round(test_problem, train_x, train_y, config, i=round)


        for i in range(new_x.shape[0]):
            # make a dict of each point and add to log
            point = {}
            point['round'] = round+1
            point['y'] = new_y[i].item()
            for j in range(new_x.shape[1]):
                point['x'+str(j)] = new_x[i,j].item()
            point['time'] = time.time()
            experiment_log.append(point)



        

        # update train_x and train_y
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

    
    # experiment done. Train a final model on all data, in case we want to use it later.
    train_x = normalize(train_x, test_problem.bounds)
    train_y = standardize(train_y)

    # print('Setting up GP')
    if config['keops']:
        kernel = get_matern_kernel_with_gamma_prior(
                    ard_num_dims=train_x.shape[-1], # BoTorch default: 1 length scale per input dim.
                    batch_shape=train_x.shape[:-2],
                )
        model = SingleTaskGP(train_x.detach(),train_y.detach(), covar_module=kernel)
    else:
        model = SingleTaskGP(train_x.detach(),train_y.detach())
    
    if torch.get_default_dtype() == torch.float64:
        model = model.double()
        model.likelihood = model.likelihood.double()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.to(device)
    mll = mll.to(device)

    try:
        mll = fit_gpytorch_mll(mll)
    except RuntimeError as e:
        print('Fitting GP failed. Retrying. with torch')
        from botorch.optim.fit import fit_gpytorch_mll_torch
        mll = fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)

    # print(f'Fitted GP on data: {train_x.shape[0]} points, {train_x.shape[1]} dimensions. Memory usage {torch.cuda.memory_allocated()/1e9} GB')

    try:
        torch.save(model, os.path.join(config['out_dir'], 'model_round_'+str(round+1)+'.pt'))
    except AttributeError as e:
        # pykeops doesn't support pickling.
        print(e)
        torch.save(model.state_dict(), os.path.join(config['out_dir'], 'model_round_'+str(round+1)+'.pt'))
        # also save train data.
        torch.save([model.train_inputs, model.train_targets], os.path.join(config['out_dir'], 'model_round_'+str(round+1)+'_train_data.pt'))


    config['run_time'] = time.time() - start_time

    # make a dataframe from the log
    # save the dataframe to a csv
    # save the config to a json    
    df = pd.DataFrame(experiment_log)
    df.to_csv(os.path.join(config['out_dir'], 'experiment_log.csv'))
    with open(os.path.join(config['out_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)




def main():

    config = {
        'test_function': 'rastrigin',
        'dim': 10,
        'seed': 0,
        'n_start_points': 100,
        'n_rounds': 20,
        'batch_size': 100,
        'num_restarts': 10,
        'raw_samples': 100,
        'acq_fn': 'beebo',
        'explore_parameter': 0.1,
        'opt': 'torch',
        'run_name': None,
        'keops': False,
        'init_min_distance': 0.5,
        'logdet_method': 'svd', 
        'n_thompson_base_samples': 10000,
        'gibbon_n_candidates': 100000,
        'custom_inference': False,
        'run_dir_prefix': 'runs',
    }
    skip=True # skip if output file already exists.

    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if type(value) == bool:
            parser.add_argument(f'--{key}', action='store_true', default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value) if value is not None else None, default=value)
    args = parser.parse_args()
    config.update(vars(args))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if config['run_name'] is None:
        if config['acq_fn'] in ['qei', 'random', 'thompson', 'gibbon', 'modifiedgibbon', 'krigingbeliever', 'jes', 'sequential_thompson']:
            # explore parameter is not used. don't include in run_name
            run_name = f'{config["test_function"]}{config["dim"]}_q{config["batch_size"]}/{config["acq_fn"]}'
        else:
            run_name = f'{config["test_function"]}{config["dim"]}_q{config["batch_size"]}/{config["acq_fn"]}_explore_parameter{config["explore_parameter"]}'
    else:
        run_name = config['run_name']

    config['out_dir'] = f'{config["run_dir_prefix"]}_{config["seed"]}/{run_name}'


    print(config['out_dir'])

    if os.path.exists(os.path.join(config['out_dir'], 'experiment_log.csv')) and skip==True:
        print('Experiment already run. Skipping.')
        return  # don't run again.

    # covar_root_decomposition=False prevents MC sampling from crashing.
    with gpytorch.settings.max_cholesky_size(2), gpytorch.settings.fast_computations(solves=True, covar_root_decomposition=False):
        run_bo_rounds(config)


if __name__ == '__main__':
    main()

