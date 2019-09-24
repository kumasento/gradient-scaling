# SSD

This directory contains code to explore the training of SSD by adaptive loss scaling.

## Changes made to the original SSD model

The original SSD example provided in [chainercv](https://github.com/chainer/chainercv/tree/master/examples/ssd) cannot be automatically transformed to one that supports adaptive loss scaling.

We've made the following changes to ensure the transformation can carry out smoothly:

1. The backbone VGG network is now built on `PickableSequentialChain`, such that the branching points among its layers can be automatically discovered and transformed to `AdaLossBranch`.
2. Convolution layers in the `Multibox` class are now initialized at the highest level, instead of being wrapped within `ChainList`, which cannot be discovered by our `AdaLossScaled` API.
3. The post-processing and concatenation are wrapped into functions and initialized within `Multibox`'s scope.
4. We need to call `AdaLossBranch` explicitly in the `forward` method of `Multibox` such that its branches can propagate gradients correctly.

## Commands

Please remember to add `mpiexec -n <N_GPU>` as a prefix if you intend to run this code in an MPI environment.

```shell
# adaptive loss scaling
python3 train_multi.py --model ssd512 --dtype mixed16 --out /home/user/results --loss-scale-method approx_range

# static loss scaling
python3 train_multi.py --model ssd512 --dtype mixed16 --out /home/user/results --loss-scale-method fixed --init-scale 128

# dynamic loss scaling
# fixed makes sure that ada_loss won't take any effect
python3 train_multi.py --model ssd512 --dtype mixed16 --out /home/user/results --loss-scale-method fixed --dynamic-interval 10

# no loss scaling
python3 train_multi.py --model ssd512 --dtype mixed16 --out /home/user/results --loss-scale-method fixed --init-scale 1
```

## Results

| Model  | Loss scale                        | Batch size | Validation mAP |
| ------ | --------------------------------- | ---------- | -------------- |
| SSD512 | FP32 baseline                     | 8          | 78.94          |
| SSD512 | adaptive                          | 8          | **0.7927**     |
| SSD512 | adaptive (freq=10)                | 8          | 0.7924         |
| SSD512 | no loss scale                     | 8          | diverged       |
| SSD512 | loss scale (8)                    | 8          | 0.7883         |
| SSD512 | loss scale (128)                  | 8          | 0.7871         |
| SSD512 | loss scale (256)                  | 8          | 0.7836         |
| SSD512 | loss scale (dynamic, interval=10) | 8          | 0.7504         |

| Model  | Loss scale                | Batch size | Validation mAP |
| ------ | ------------------------- | ---------- | -------------- |
| SSD512 | adaptive                  | 32         | 0.8030         |
| SSD512 | no loss scale             | 32         | diverged       |
| SSD512 | loss scale (128)          | 32         | 0.8001         |
| SSD512 | loss scale (dyn, intv=10) | 32         | 0.8017         |

## TODO

- [ ] Reduce memory usage by fully transforming `ReLU`.
- [ ] Remove the explicit reference of `AdaLossBranch` in the SSD model (`Multibox`).
- [ ] Support cuDNN
