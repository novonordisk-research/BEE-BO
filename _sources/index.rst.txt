.. BEEBO documentation master file, created by
   sphinx-quickstart on Mon Jul 15 13:44:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BEEBO documentation
===================

BEEBO is a family of acquisition functions for Bayesian optimization that natively scale to batched acquisition
and trade off exploration and exploitation explicitly. The :mod:`~beebo` package is compatible with BoTorch. 

Installation
------------

To install BEEBO, you can use pip:

.. code-block:: bash

   pip install beebo

   

Usage
-----

Both meanBEEBO and maxBEEBO are implemented as the :class:`~beebo.acquisition.BatchedEnergyEntropyBO` class.

To use :class:`~beebo.acquisition.BatchedEnergyEntropyBO`, you need to follow these steps:

1. Import the class:
   
   .. code-block:: python
     
     from beebo import BatchedEnergyEntropyBO

2. Create an instance of :class:`~beebo.acquisition.BatchedEnergyEntropyBO`:
   
   .. code-block:: python

      # you need to set up a GP model first. Nothing special here - just a standard BoTorch setup.
     
     amplitude = model.covar_module.outputscale.item() # get the GP's kernel amplitude

     beebo = BatchedEnergyEntropyBO(
      model, # a gaussian process model.
      temperature=1.0, 
      kernel_amplitude=amplitude, # used for scaling the temperature.
      energy_function='sum', # "sum" for meanBEEBO, "softmax" for maxBEEBO
      logdet_method='svd', # LinAlg: how to compute log determinants
      augment_method='naive', # LinAlg: how to perform the train data augmentation
      )
   

3. Use BoTorch to optimize the acquisition function. Standard BoTorch parameters need to be set.
   
   .. code-block:: python
     
     from botorch.optim.optimize import optimize_acqf
     
     points, value = optimize_acqf(
      acq_fn, 
      q=100, # the batch size     
      bounds=bounds, # the bounds of the optimization problem
      # botorch hyperparameters for optimization
      num_restarts=10, 
      raw_samples=100, 
      )
     





.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   beebo
   modules
   

