# BEE-BO
Batched Energy-Entropy acquisition for Bayesian Optimization


![bee-bo](https://github.com/fteufel/BEE-BO/assets/56223326/f179d070-b089-4aeb-b2ce-dce9020ed63a|width=20)


This repository provides a BoTorch/GPyTorch implementation of the BEE-BO acquisition function for Bayesian Optimization.


## Installation

To install the package, clone the repository and run

```
pip install -e .
```

## Usage

The BEE-BO acquisition function is fully compatible with BoTorch and is implemented as a `AnalyticAcquisitionFunction`. It can be used as follows, using standard BoTorch utilities. 

```python
from beebo import BatchedEnergyEntropyBO


amplitude = model.covar_module.outputscale.item() # get the GP's kernel amplitude

acq_fn = BatchedEnergyEntropyBO(
    model, # a gaussian process model.
    temperature=1.0, 
    kernel_amplitude=amplitude)

points, value = optimize_acqf(
    acq_fn, 
    q=100, 
    num_restarts=10, 
    bounds=bounds, # the bounds of the optimization problem
    raw_samples=100, 
    )
```
For setting up `model` and `bounds`, please refer to BoTorch's tutorials.

### Hyperparameters

The explore-exploit trade-off of BEE-BO is controlled using its temperature parameter. We additionally use the kernel amplitude to scale the temperature internally, so that it is comparable to the `beta` parameter in the standard Upper Confidence Bound (UCB) acquisition function. When the `kernel_amplitude` is 1.0, the scaling has no effect and you recover the "pure" BEE-BO acquisition function (i.e. `A=sum(posterior_mean)+temparature*information_gain`).


## Experiments

Please see the `benchmark` directory for the code to perform the benchmark experiments from the paper.

## Implementation notes

## Implementation notes

BEE-BO requires temporarily adding the query data points as train data points to the GP model in the forward pass to compute the information gain. GPyTorch offers some functionality for that, such as `set_train_data` or `get_fantasy_model`. In our experiments, both these approaches resulted in memory leaks when running with gradients enabled. As a workaround, we duplicate the GP model via deepcopy, and then set the train data of the duplicated GP before calling it. This, together with adding the posterior mean multiplied by 0 to the result, seems to be the only way to avoid memory leaks as of GPyTorch version 1.11 .

The forward method thus may look a bit convoluted. The methods `compute_energy` and `compute_entropy` are not used for above reasons, but show the core of the BEE-BO algorithm in a more readable way.
