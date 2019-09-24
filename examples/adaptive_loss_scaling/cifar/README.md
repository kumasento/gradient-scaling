# Train CNN on CIFAR by Adaptive Loss Scaling

This directory contains CNN training examples on CIFAR-10/100. For now we mainly support training ResNet variants.

Training and visualization are implemented in `utils.py` as `train` and `plot` respectively. The base ResNet model are implemented as `ResNetCIFAR` under a `chainerlp` module `chainerlp.links.model.resnet`. ResNet models will be transformed to support adaptive loss scaling by applying `AdaLossTransformLinear`, `AdaLossTransformConv2DBNActiv`, and `AdaLossTransformBasicBlock`, the last two of which are customized for `chainerlp`.

As the MNIST example, we fix the data type used for training to `chainer.mixed16` and we apply `use_fp32_update` to optimizers. Also, you can set `update_per_n_iteration` to control the frequency of loss scale calculation. A recommended value is `1000`.

## Usage

`train_resnet_on_cifar.py` provides an easy-to-use interface for training.

```shell
python train_resnet_on_cifar.py --n_layer 110 --n_class 10 --gpu 0 --manual_seed 0 --out /tmp/result --update_per_n_iteration 1000
```

The command above trains a ResNet-110 model on CIFAR-10 by GPU (ID: 0) with a random seed 0. Results will be stored under `/tmp/result`. Adaptive loss scale will be calculated per 1000 iterations.

`run_train.sh` summarizes how we train the models listed in the paper.