ImageNet Training
===

This directory contains the ImageNet training example.

## Commands

The training command:

```shell
mpiexec -n <number of GPU> python train_imagenet_multi.py --dataset-dir <ILSVRC2012 directory> -a <resnet18,resnet50> --dtype mixed16 --out <result directory> --loss-scale-method approx_range
```

Code to reproduce results:

```shell
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet18 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet18 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range --update-per-n-iteration 1000
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet18 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method fixed --init-scale 128
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet18 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method fixed --init-scale 1
# dynamic loss scaling
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet18 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method fixed --dynamic-interval 10

python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range --update-per-n-iteration 1000
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method fixed --init-scale 128
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method fixed --init-scale 1
# dynamic loss scaling
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method fixed --dynamic-interval 10

# evaluating accum bound
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range --update-per-n-iteration 1000 --accum-upper-bound 2048
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range --update-per-n-iteration 1000 --accum-upper-bound 4096
# evaluating scale bound
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range --update-per-n-iteration 1000 --scale-upper-bound 128 
python3 examples/adaptive_loss_scaling/imagenet/train_imagenet_multi.py -a resnet50 --dataset-dir /home/user/data/imagenet --dtype mixed16 --out /home/user/results --loss-scale-method approx_range --update-per-n-iteration 1000 --scale-upper-bound 1024
```

## Results

The way to get these data has been logged in the [Analysis ResNet Result](./Analysis_ResNet_Result.ipynb) notebook.
