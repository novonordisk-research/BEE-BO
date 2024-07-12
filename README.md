# BEEBO
**Batched Energy-Entropy acquisition for Bayesian Optimization**

![bee-bo small](https://github.com/fteufel/BEE-BO/assets/56223326/e7b9490c-cd65-4598-a55c-5023ced3ce33)



This repository provides a BoTorch/GPyTorch implementation of the BEEBO acquisition function for batched Bayesian Optimization.



## Installation

To install the package, run

```
pip install beebo
```

Alternatively, clone the repository and run `pip install -e.`

## Usage

The BEE-BO acquisition function is fully compatible with BoTorch and is implemented as an [`AnalyticAcquisitionFunction`](https://botorch.org/api/acquisition.html#analytic-acquisition-function-api). It can be used as follows, using [standard BoTorch utilities](https://botorch.org/docs/getting_started). 

```python
from beebo import BatchedEnergyEntropyBO
from botorch.optim.optimize import optimize_acqf

# `model` is e.g. a SingleTaskGP trained according to BoTorch's tutorials
# `bounds` is the search space of the optimization problem

amplitude = model.covar_module.outputscale.item() # get the GP's kernel amplitude

acq_fn = BatchedEnergyEntropyBO(
    model, # a gaussian process model.
    temperature=1.0, 
    kernel_amplitude=amplitude,
    energy_function='sum', # "sum" for meanBEEBO, "softmax" for maxBEEBO
    logdet_method='svd', # LinAlg: how to compute log determinants
    augment_method='naive', # LinAlg: how to perform the train data augmentation
    )

points, value = optimize_acqf(
    acq_fn, 
    q=100, 
    num_restarts=10, 
    bounds=bounds, # the bounds of the optimization problem
    raw_samples=100, 
    )
```
For setting up `model` and `bounds`, please refer to [BoTorch's tutorials](https://botorch.org/tutorials/).

### Hyperparameters

The explore-exploit trade-off of BEE-BO is controlled using its temperature parameter. In the code snippet above, we additionally use the kernel amplitude (output scale) to scale the temperature internally, so that it is comparable to the `beta` parameter in the standard Upper Confidence Bound (UCB) acquisition function. When the `kernel_amplitude` is 1.0, the scaling has no effect and you recover the "pure" BEE-BO acquisition function, 

$a(\mathbf{x})=-E(\mathbf{x})+T*I(\mathbf{x})$.


## Experiments

Please see the `benchmark` directory in the repository for the code to perform the benchmark experiments from the paper.



### Configuring the linear algebra routines for BEE-BO

- Log determinants:

   The `BatchedEnergyEntropyBO` constructor accepts a `logdet_method` argument, which can be set to `svd`, `cholesky` ot `torch`.   
   - The default is `svd`, which is stable, at the expense of being slow.  
   - `cholesky` exploits the PSD structure by computing the log determinant from a cholesky decomposition, which is fast but can suffer from numerical instability.  
   - `torch` simply uses PyTorch's `logdet` function, which can also suffer from numerical instability.

- GP augmentation with new training observations:

    The `BatchedEnergyEntropyBO` constructor accepts a `augment_method` argument, which can be set to `naive`, `cholesky` or `get_fantasy_model`.  
    - The default is `naive`, which simply keeps a second GP and calls `set_train_data` with augmented tensors. This uses default GPytorch infrastructure, does not suffer from any memory leaks, but requires recomputing all caches.  
    - `cholesky` performs a low rank update to the precomputed cholesky decomposition of the train-train covariance, and then computes the posterior covariance via a cholesky solve. In our experiments, this was faster than both the naive approach and the LOVE low-rank update.
    - `get_fantasy_model` uses GPyTorch's `get_fantasy_model` method, which can perform a low rank update when using LOVE, but suffers from a memory leak when using gradients. Also, the low rank update is contingent on a pending patch at https://github.com/cornellius-gp/gpytorch/pull/2494.


## Implementation notes

Log determinants are computed using singular value decomposition (SVD) for numerical stability.

BEE-BO requires temporarily adding the query data points as training data points to the GP model in the forward pass to compute the information gain. GPyTorch offers some functionality for that, such as `set_train_data` or `get_fantasy_model`. In our experiments with GPyTorch 1.11, both these approaches resulted in memory leaks when running with gradients enabled. As a workaround, we duplicate the GP model via deepcopy when initializing the acquisition function, and then set the train data of the duplicated GP before calling it to compute the augmented posterior. This, together with adding the posterior mean multiplied by 0 to the result, seems to be avoiding memory leaks for the current version.  
Due to these workarounds, the forward method thus may look a bit convoluted. The methods `compute_energy` and `compute_entropy` are not used for above reasons, but show the core operations of the BEE-BO algorithm in a more readable way.