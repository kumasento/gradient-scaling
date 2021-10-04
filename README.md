# Reducing Underflow in Mixed Precision Training by Gradient Scaling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project implements the _gradient scaling_ method to improve the performance of [mixed precision training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

The old repository: https://github.com/ada-loss/ada-loss

```bibtex
@inproceedings{ijcai2020-404,
  title     = {Reducing Underflow in Mixed Precision Training by Gradient Scaling},
  author    = {Zhao, Ruizhe and Vogel, Brian and Ahmed, Tanvir and Luk, Wayne},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {2922--2928},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/404},
  url       = {https://doi.org/10.24963/ijcai.2020/404},
}
```

## Introduction

Loss scaling is a technique that scales up loss values to mitigate underflow caused by low precision data representation in backpropagated activation gradients. The original implementation uses a fixed loss scale value predetermined before training starts for all layers, which may not be optimal since the statistics of gradients change across layers and training epochs. Instead, our method calculates the loss scale value for each layer based on their runtime statistics.

## Installation

We are using Anaconda to manage package dependencies:

```shell
conda create -f environment.yml
conda activate ada_loss
```

To install this project, please consider using this command:

```shell
pip install -e . # in the project root
```

## Project structure

The structure of this project is as follows: the core of the adaptive loss scaling method is implemented in the `ada_loss` package; `chainerlp` provides the implementation of some baseline models; and `models` includes third party implementation of more complicated baseline models.

## Usage

Example usage for `chainer` (other frameworks will be released later):

```python
from ada_loss.chainer import AdaLossScaled
from ada_loss.chainer import transforms

# transform your link to support adaptive loss scaling
link = AdaLossScaled(link, transforms=[
    transforms.AdaLossTransformLinear(),
    transforms.AdaLossTransformConvolution2D(),
    # ...
])
```

It tries to convert links within the given `link` to ones that supports adaptive loss scaling based on the provided list of `transforms`. Adaptive loss scaled links are located under `ada_loss.chainer.links`. Transforms are extended based on `AdaLossTransform` in `ada_loss.chainer.transforms.base` and stored under `ada_loss.chainer.transforms`. For now, users are required to go through their link and specify explicitly transforms that should be taken.

## Examples

Examples are located [here](examples/adaptive_loss_scaling).

## Testing

Tests can be launched by calling `pytest`. Some tests are specified to be run on GPUs.
