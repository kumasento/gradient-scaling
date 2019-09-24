# Adaptive Loss Scaling Implementation in Chainer

This module provides the implementation of adaptive loss scaling specific to `chainer`.

- [Adaptive Loss Scaling Implementation in Chainer](#adaptive-loss-scaling-implementation-in-chainer)
  - [Main class](#main-class)
  - [Calculate Loss Scale](#calculate-loss-scale)
  - [Propagate Loss Scale](#propagate-loss-scale)
  - [Transform Links](#transform-links)

## Main class

`AdaLossChainer` provides the `chainer` specific implementation of `AdaLoss`.

## Calculate Loss Scale

Loss scale calculation is implemented in the following functions:

- `get_loss_scale_by_approx_range`
- `get_loss_scale_by_element_wise_range`
- `get_loss_scale_by_abs_range`

All functions should return a 0-dim `numpy` array. Computation is currently implemented as CuPy functions on GPU.

After collecting these "unbound" loss scale values, we apply further heuristic constraints on the loss scale values, which are implemented in `bound_loss_scale_by_heuristics`. We mainly bound the loss scale value by `scale_upper_bound` and accumulated loss scale by `accum_upper_bound`.

## Propagate Loss Scale

Calculated loss scale is attached to gradient `Variable` as an attribute `loss_scale`. Attaching loss scale is performed in `AdaLoss.loss_scaling`, which means that if you calculate the scale through `AdaLoss` and its inheritance during the backward pass, loss scale values will be definitely attached.

But there might be functions in a model that don't calculate loss scale, e.g., ReLU, which hinders loss scale propagating. Our solution is to wrap these functions with `IdentityLossScalingWrapper` defined [here](link/identity_loss_scaling.py), which helps loss scale bypass those functions.

## Transform Links

The `AdaLossScaled` API defined [here](ada_loss_scaled.py) transforms your link to support adaptive loss scaling, mainly by replacing links and wrapping functions. Links will be replaced by their counterparts, e.g., `AdaLossLinear` and `AdaLossConvolution2D`. Functions will be wrapped by `IdentityLossScalingWrapper`.

This API is risky to use. Please print out the transformed link before using it.
