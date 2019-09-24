# Adaptive Loss Scaling Implementation

This document explains the core ideas behind the implementation of our adaptive loss scaling method.

Recap: adaptive loss scaling assigns a loss scale to each layer, which is computed by runtime values to minimize the amount of underflow activation gradients during backpropagation.

- [Adaptive Loss Scaling Implementation](#adaptive-loss-scaling-implementation)
  - [Design Principles](#design-principles)
  - [Project structure](#project-structure)
  - [Classes](#classes)
  - [Configuration](#configuration)
  - [Loss Scale Calculation](#loss-scale-calculation)
    - [Fixed Loss Scale (`fixed`)](#fixed-loss-scale-fixed)
    - [Approximated by Statistics (`approx_range`)](#approximated-by-statistics-approxrange)
    - [Element-wise (`element_wise_range`)](#element-wise-elementwiserange)
    - [Min-Max (`abs_range`)](#min-max-absrange)
  - [Transformation](#transformation)
  - [Debugging](#debugging)

## Design Principles

- **Flexibility** for configuration: parameters that configure the algorithm should be easily adjusted in a plugin style.
- **Extensibility** to other frameworks: we try to separate out the implementation that is specific to a framework.

## Project structure

We will place codes that are common for all frameworks in the root directory, and framework-specific code in their own modules, e.g., [`chainer`](chainer).

## Classes

[`AdaLoss`](ada_loss.py) is the base class that configures and implements adaptive loss scaling. [`AdaLossChainer`](chainer/ada_loss.py) inherits it and implements its abstract methods in `chainer`. For `chainer` specific classes, please go to [`chainer`](chainer).

## Configuration

We have the following categories of configuration:

- **General** configurations: `u_min` and `u_max` that provide the lower and upper bounds of the representable range; `recorder` takes the instance that records loss scale values;
- **AdaLoss** general configuration for all adaptive loss scaling methods: `loss_scale_method` specifies which method we want to use; `update_per_n_iteration` indicates the loss scale calculation frequency; `debug_level` provides the degree of debugging; `init_scale` gives the starting loss scale before backpropagation happens.
- **Method** specific configuration: `n_uf_threshold` is the threshold that we want to constrain the amount of underflow values in the `approx_range` method;

## Loss Scale Calculation

There are three different ways to calculate adaptive loss scales.

### Fixed Loss Scale (`fixed`)

We assign a loss scale value that will be used for all layers by `fixed_loss_scale`. For example, if you want to scale the loss by a fixed value `16`, you can set `init_scale` to `16` and `fixed_loss_scale` to 1. Note that you can't set `fixed_loss_scale` to `16` since the actual loss scale value for each layer is accumulated during backpropagation.

### Approximated by Statistics (`approx_range`)

This is the default method: we calculate the loss scale by statistics of gradients and weights.

### Element-wise (`element_wise_range`)

This method calculates the exact range of product values between gradients and weights, and use it to find the loss scale.

### Min-Max (`abs_range`)

It is an improved version of `element_wise_range`. Instead of iterating every element, this method only uses the minimum and maximum values of gradients and weights to compute a wider range of product values.

## Transformation

Transformation converts a model to support adaptive loss scaling.

## Debugging

We also provide some debugging utilities to record loss scale values.
