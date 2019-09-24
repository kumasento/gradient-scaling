""" Train ImageNet on a multi-node environment.

Ref:
[1] https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf
[2] https://github.com/xu3kev/imagenet_dmux/blob/master/train_imagenet_multi.py
"""

import os
import argparse
import multiprocessing
from contextlib import ExitStack
import numpy as np

import chainer
from chainer import iterators
import chainer.links as L
from chainer import optimizer
from chainer import optimizers
from chainer import training
from chainer import serializers
from chainer.training import extensions
from chainer.optimizer_hooks import WeightDecay

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv import datasets
from chainercv import transforms

# NOTE: buggy
# from chainercv.chainer_experimental.training.extensions import make_shift

import chainermn

# ChainerLP
from chainerlp import utils
from chainerlp import visualize
from chainerlp import snapshot
from chainerlp.links import models
from chainerlp.links.models.resnet import Bottleneck
from chainerlp.extensions import make_shift
from chainerlp.hooks import AdaLossMonitor
from chainerlp.ada_loss.transforms import *

# AdaLoss support
from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer_impl.ada_loss import AdaLossChainer
from ada_loss.chainer_impl.ada_loss_transforms import *
from ada_loss.chainer_impl.ada_loss_recorder import AdaLossRecorder

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and
                     callable(models.__dict__[name]))

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def set_random_seed(args, device):
    # Set up random seed
    if args.manual_seed is not None:
        utils.set_random_seed(args.manual_seed, device=device)


class TrainTransform(object):

    def __init__(self, mean, args):
        self.mean = mean
        self.args = args

    def __call__(self, in_data):
        img, label = in_data
        img = transforms.random_sized_crop(img)
        img = transforms.resize(img, (224, 224))
        img = transforms.random_flip(img, x_random=True)
        img -= self.mean
        return img.astype(chainer.get_dtype()), label


class ValTransform(object):

    def __init__(self, mean, args):
        self.mean = mean
        self.args = args

    def __call__(self, in_data):
        img, label = in_data
        img = transforms.scale(img, 256)
        img = transforms.center_crop(img, (224, 224))
        img -= self.mean
        return img.astype(chainer.get_dtype()), label


# data types
dtypes = {
    'float16': np.float16,
    'float32': np.float32,
    'mixed16': chainer.mixed16,
}

parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--dataset-dir', help='Path to the ILSVRC2012 dataset')
parser.add_argument('--arch',
                    '-a',
                    choices=model_names,
                    default='resnet50',
                    help='Supported models')
parser.add_argument('--dtype',
                    choices=dtypes.keys(),
                    type=str,
                    default='float32',
                    help='Data type')
parser.add_argument('--communicator',
                    type=str,
                    default='pure_nccl',
                    help='Type of communicator')
parser.add_argument('--loaderjob', type=int, default=4)
parser.add_argument('--batchsize',
                    type=int,
                    default=32,
                    help='Batch size for each worker')
parser.add_argument('--lr', type=float)
parser.add_argument('--lr-decay',
                    type=float,
                    default=0.1,
                    help='Learning rate decay value')
parser.add_argument('--weight-decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay value (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--out',
                    type=str,
                    default='results',
                    help='Where to place the training result')
parser.add_argument('--epoch', type=int, default=90)
parser.add_argument('--iter',
                    type=int,
                    default=None,
                    help='Number of iterations to train (default: None)')
parser.add_argument('--snapshot-freq',
                    type=int,
                    default=1,
                    help='Frequency to save snapshots')
parser.add_argument('--manual-seed',
                    default=None,
                    type=int,
                    help='Default random seed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Verbose output')
parser.add_argument('--resume',
                    type=str,
                    default=None,
                    help='Specify the snapshot file to be resumed from')

# Additional configurations
parser.add_argument('--first-bn-mixed16',
                    action='store_true',
                    default=False,
                    help='Should use mixed16 for the first BN layer')

# ada-loss configuration
parser.add_argument('--init-scale',
                    default=1,
                    type=float,
                    help='Initial scale for ada loss')
parser.add_argument('--loss-scale-method',
                    default='abs_range',
                    type=str,
                    help='Method for adaptive loss scaling')
parser.add_argument('--scale-upper-bound',
                    default=16,
                    type=float,
                    help='Hard upper bound for each scale factor')
parser.add_argument('--accum-upper-bound',
                    default=1024,
                    type=float,
                    help='Accumulated upper bound for all scale factors')
parser.add_argument('--update-per-n-iteration',
                    default=1,
                    type=int,
                    help='Update the loss scale value per n iteration')
parser.add_argument('--dynamic-interval',
                    default=None,
                    type=int,
                    help='Interval for dynamic loss scaling')

args = parser.parse_args()

# Set up data type
chainer.global_config.dtype = dtypes[args.dtype]
# # Disable tensorcore
# chainer.global_config.use_cudnn_tensor_core = 'never'

# ImageNet mean
_mean = [123.15163084, 115.90288257, 103.0626238]
_mean = np.array(_mean)[:, np.newaxis, np.newaxis].astype(chainer.get_dtype())


def main():
    # Start the multiprocessing environment
    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    # Set up workspace
    # 12 GB GPU RAM for workspace
    chainer.cuda.set_max_workspace_size(16 * 1024 * 1024 * 1024)

    # Setup the multi-node environment
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank
    print(
        '==> Successfully setup communicator: "{}" rank: {} device: {} size: {}'
        .format(args.communicator, comm.rank, device, comm.size))
    set_random_seed(args, device)

    # Setup LR
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 0.1 * (args.batchsize * comm.size) / 256  # TODO: why?
        if comm.rank == 0:
            print('LR = {} is selected based on the linear scaling rule'.format(
                lr))

    # Setup dataset
    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'val')
    label_names = datasets.directory_parsing_label_names(train_dir)
    train_data = datasets.DirectoryParsingLabelDataset(train_dir)
    val_data = datasets.DirectoryParsingLabelDataset(val_dir)
    train_data = TransformDataset(train_data, ('img', 'label'),
                                  TrainTransform(_mean, args))
    val_data = TransformDataset(val_data, ('img', 'label'),
                                ValTransform(_mean, args))
    print('==> [{}] Successfully finished loading dataset'.format(comm.rank))

    # Initializing dataset iterators
    if comm.rank == 0:
        train_indices = np.arange(len(train_data))
        val_indices = np.arange(len(val_data))
    else:
        train_indices = None
        val_indices = None

    train_indices = chainermn.scatter_dataset(train_indices, comm, shuffle=True)
    val_indices = chainermn.scatter_dataset(val_indices, comm, shuffle=True)
    train_data = train_data.slice[train_indices]
    val_data = val_data.slice[val_indices]
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, n_processes=args.loaderjob)
    val_iter = iterators.MultiprocessIterator(val_data,
                                              args.batchsize,
                                              repeat=False,
                                              shuffle=False,
                                              n_processes=args.loaderjob)

    # Create the model
    kwargs = {}
    if args.first_bn_mixed16 and args.dtype == 'float16':
        print('==> Setting the first BN layer to mixed16')
        kwargs['first_bn_mixed16'] = True

    # Initialize the model
    net = models.__dict__[args.arch](n_class=len(label_names), **kwargs)
    # Following https://arxiv.org/pdf/1706.02677.pdf,
    # the gamma of the last BN of each resblock is initialized by zeros.
    for l in net.links():
        if isinstance(l, Bottleneck):
            l.conv3.bn.gamma.data[:] = 0

    # Apply ada loss transform
    recorder = AdaLossRecorder(sample_per_n_iter=100)
    # Update the model to support AdaLoss
    net = AdaLossScaled(
        net,
        init_scale=args.init_scale,
        cfg={
            'loss_scale_method': args.loss_scale_method,
            'scale_upper_bound': args.scale_upper_bound,
            'accum_upper_bound': args.accum_upper_bound,
            'update_per_n_iteration': args.update_per_n_iteration,
            'recorder': recorder,
        },
        transforms=[
            AdaLossTransformLinear(),
            AdaLossTransformBottleneck(),
            AdaLossTransformBasicBlock(),
            AdaLossTransformConv2DBNActiv(),
        ],
        verbose=args.verbose)

    if comm.rank == 0:  # print network only in the 1-rank machine
        print(net)
    net = L.Classifier(net)
    hook = AdaLossMonitor(sample_per_n_iter=100,
                          verbose=args.verbose,
                          includes=['Grad', 'Deconvolution'])

    # Setup optimizer
    optim = chainermn.create_multi_node_optimizer(
        optimizers.CorrectedMomentumSGD(lr=lr, momentum=args.momentum), comm)
    if args.dtype == 'mixed16':
        print('==> Using FP32 update for dtype=mixed16')
        optim.use_fp32_update()  # by default use fp32 update

        # HACK: support skipping update by existing loss scaling functionality
        if args.dynamic_interval is not None:
            optim.loss_scaling(interval=args.dynamic_interval, scale=None)
        else:
            optim.loss_scaling(interval=float('inf'), scale=None)
            optim._loss_scale_max = 1.0 # to prevent actual loss scaling

    optim.setup(net)

    # setup weight decay
    for param in net.params():
        if param.name not in ('beta', 'gamma'):
            param.update_rule.add_hook(WeightDecay(args.weight_decay))

    # allocate model to multiple GPUs
    if device >= 0:
        chainer.cuda.get_device(device).use()
        net.to_gpu()

    # Create an updater that implements how to update based on one train_iter input
    updater = chainer.training.StandardUpdater(train_iter, optim, device=device)
    # Setup Trainer
    stop_trigger = (args.epoch, 'epoch')
    if args.iter is not None:
        stop_trigger = (args.iter, 'iteration')
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    @make_shift('lr')
    def warmup_and_exponential_shift(trainer):
        """ LR schedule for training ResNet especially.
        NOTE: lr should be within the context.
        """
        epoch = trainer.updater.epoch_detail
        warmup_epoch = 5  # NOTE: mentioned the original ResNet paper.
        if epoch < warmup_epoch:
            if lr > 0.1:
                warmup_rate = 0.1 / lr
                rate = warmup_rate \
                    + (1 - warmup_rate) * epoch / warmup_epoch
            else:
                rate = 1
        elif epoch < 30:
            rate = 1
        elif epoch < 60:
            rate = 0.1
        elif epoch < 80:
            rate = 0.01
        else:
            rate = 0.001
        return rate * lr

    trainer.extend(warmup_and_exponential_shift)
    evaluator = chainermn.create_multi_node_evaluator(
        extensions.Evaluator(val_iter, net, device=device), comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)

        # NOTE: may take snapshot every iteration now
        snapshot_label = 'epoch' if args.iter is None else 'iteration'
        snapshot_trigger = (args.snapshot_freq, snapshot_label)
        snapshot_filename = ('snapshot_' + snapshot_label + '_{.updater.' +
                             snapshot_label + '}.npz')
        trainer.extend(extensions.snapshot(filename=snapshot_filename),
                       trigger=snapshot_trigger)

        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_value(
            'loss_scale',
            lambda trainer: trainer.updater.get_optimizer('main')._loss_scale),
                       trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'iteration', 'epoch', 'elapsed_time', 'lr', 'loss_scale', 'main/loss',
            'validation/main/loss', 'main/accuracy', 'validation/main/accuracy'
        ]),
                       trigger=print_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    recorder.trainer = trainer
    hook.trainer = trainer
    with ExitStack() as stack:
        if comm.rank == 0:
            stack.enter_context(hook)
        trainer.run()

    # store recorded results
    if comm.rank == 0:  # NOTE: only export in the first rank
        recorder.export().to_csv(os.path.join(args.out, 'loss_scale.csv'))
        hook.export_history().to_csv(os.path.join(args.out, 'grad_stats.csv'))


if __name__ == '__main__':
    main()
