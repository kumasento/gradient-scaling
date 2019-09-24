""" Utility functions for training models on specific datasets """
import shutil
import tempfile
from contextlib import ExitStack

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I

from chainer import training
from chainer import dataset
from chainer.training import extensions

from chainercv import transforms

import chainerlp
import chainerlp.extensions
from chainerlp import notebook_utils
from chainerlp import utils
from chainerlp.links import models


def train_model_on_mnist(net,
                         get_mnist=None,
                         epoch=100,
                         schedule=None,
                         lr_decay=0.1,
                         batchsize=64,
                         learnrate=0.1,
                         device=-1,
                         sample_iterations=None,
                         loss_scale=1.0,
                         cleanup=True,
                         hooks=None,
                         ndim=1,
                         recorder=None,
                         **kwargs):
    """ Train the given model on MNIST """
    # Model
    model = L.Classifier(net)
    model.to_device(device)

    # Dataset
    if get_mnist is None:
        get_mnist = chainer.datasets.get_mnist

    train, test = get_mnist(ndim=ndim)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test,
                                                 batchsize,
                                                 repeat=False,
                                                 shuffle=False)

    # Optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=learnrate)
    optimizer.setup(model)
    optimizer.use_fp32_update()
    # optimizer.loss_scaling(interval=float('inf'), scale=None)
    # optimizer._loss_scale_max = 1.0 # to prevent actual loss scaling
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))

    # Set up a trainer
    out = tempfile.mkdtemp(prefix='mnist_train-')
    print('==> Storing temporary results to {} ...'.format(out))
    updater = training.updaters.StandardUpdater(train_iter,
                                                optimizer,
                                                device=device,
                                                loss_scale=loss_scale)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    # logging
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'lr',
            'main/loss',
            'validation/main/loss',
            'main/accuracy',
            'validation/main/accuracy',
            'elapsed_time',
        ]))
    if schedule is not None:
        trainer.extend(training.extensions.ExponentialShift(
            'lr', lr_decay),
                       trigger=training.triggers.ManualScheduleTrigger(schedule, 'epoch'))

    if recorder is not None:
        recorder.setup(trainer)

    # Run
    if hooks is None:
        hooks = []

    with ExitStack() as stack:
        for hook in hooks:
            if hasattr(hook, 'trainer'):
                hook.trainer = trainer  # patch the hooks
            stack.enter_context(hook)
        trainer.run()
    log = notebook_utils.load_train_log(train_dir=out)

    if cleanup:
        print('==> Cleaning up {} ...'.format(out))
        shutil.rmtree(out)

    return hooks, log


class PreprocessCIFARTrainData(dataset.DatasetMixin):
    """ Data augmentation for CIFAR-10/100 data based on:
    https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py
    https://blog.shikoan.com/chainer-preprocess-datasetminix/
    """

    def __init__(self, pairs, mean, std):
        self.pairs = pairs
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.pairs)

    def get_example(self, i):
        """ Pad input with length 4, random crop to 32, together with random horizontal flipping. """
        x, y = self.pairs[i]
        # label
        y = np.array(y, dtype=np.int32)

        # padding with length-4 zeros
        pad_x = np.zeros((3, 40, 40), dtype=x.dtype)
        pad_x[:, 4:36, 4:36] = x
        # random cropping
        x = transforms.random_crop(pad_x, (32, 32))
        # random horizontal flipping
        x = transforms.random_flip(x, x_random=True)
        # normalize
        x = (x - self.mean) / self.std

        return x, y


class PreprocessCIFARTestData(dataset.DatasetMixin):
    """ Data augmentation for CIFAR-10/100 data """

    def __init__(self, pairs, mean, std):
        self.pairs = pairs
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.pairs)

    def get_example(self, i):
        """ Simply normalize """
        x, y = self.pairs[i]
        # label
        y = np.array(y, dtype=np.int32)
        # normalize
        x = (x - self.mean) / self.std

        return x, y


def train_model_on_cifar(net,
                         dataset='cifar10',
                         n_epoch=164,
                         batchsize=128,
                         device=-1,
                         learnrate=0.1,
                         lr_decay=0.1,
                         schedule=None,
                         weight_decay=1e-4,
                         manual_seed=None,
                         warmup_attr_ratio=None,
                         warmup_n_epoch=None,
                         cleanup=True,
                         tmpdir=None,
                         recorder=None,
                         hooks=None):
    """ Train a model on the cifar dataset """
    # Mean and Std
    _mean = np.array([0.4914, 0.4822, 0.4465],
                     dtype=chainer.get_dtype()).reshape([3, 1, 1])
    _std = np.array([0.2023, 0.1994, 0.2010],
                    dtype=chainer.get_dtype()).reshape([3, 1, 1])
    # Set up random seed
    if manual_seed is not None:
        utils.set_random_seed(manual_seed, device=device)

    if dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = chainer.datasets.get_cifar10()
        mean = _mean
        std = _std
    elif dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = chainer.datasets.get_cifar100()
        mean = np.array([0.5071, 0.4867, 0.4408],
                        dtype=chainer.get_dtype()).reshape([3, 1, 1])
        std = np.array([0.2675, 0.2565, 0.2761],
                       dtype=chainer.get_dtype()).reshape([3, 1, 1])
    else:
        raise RuntimeError('Invalid dataset choice.')

    train = PreprocessCIFARTrainData(train, mean=mean, std=std)
    test = PreprocessCIFARTestData(test, mean=mean, std=std)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test,
                                                 batchsize,
                                                 repeat=False,
                                                 shuffle=False)

    # Model initialisation
    model = L.Classifier(net)
    model.to_device(device)

    # Create optimizer
    # NOTE: here the momentum is 0.9 by default
    if warmup_attr_ratio is not None:
        learnrate *= warmup_attr_ratio
    optimizer = chainer.optimizers.MomentumSGD(learnrate)

    if chainer.get_dtype() == chainer.mixed16:
        print('==> Using FP32 update for dtype=mixed16')
        optimizer.use_fp32_update()  # by default use fp32 update
        # TODO: loss scaling

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(weight_decay))

    # Setting up the trigger for stopping training
    stop_trigger = (n_epoch, 'epoch')

    # Set up a trainer
    if tmpdir is None:
        tmpdir = '/tmp'
    out = tempfile.mkdtemp(prefix='{}_train-'.format(dataset), dir=tmpdir)
    updater = training.updaters.StandardUpdater(train_iter,
                                                optimizer,
                                                device=device)
    trainer = training.Trainer(updater, stop_trigger, out=out)

    if recorder is not None:
        recorder.setup(trainer)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    trainer.extend(extensions.observe_lr())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'lr',
            'main/loss',
            'validation/main/loss',
            'main/accuracy',
            'validation/main/accuracy',
            'elapsed_time',
        ]))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}', snapshot_on_error=True),
                   trigger=(1, 'epoch'))
    lr_shift = chainerlp.extensions.ExponentialShift(
        'lr',
        lr_decay,
        warmup_attr_ratio=warmup_attr_ratio,
        warmup_n_epoch=warmup_n_epoch,
        schedule=schedule)
    trainer.extend(lr_shift, trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # RUN
    if hooks is None:
        hooks = []
    with ExitStack() as stack:
        for hook in hooks:
            if hasattr(hook, 'trainer'):
                hook.trainer = trainer  # patch the hooks
            stack.enter_context(hook)
        trainer.run()
    log = notebook_utils.load_train_log(train_dir=out)

    if cleanup:
        print('==> Cleaning up {} ...'.format(out))
        shutil.rmtree(out)

    return hooks, log
