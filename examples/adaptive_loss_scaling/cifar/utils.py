""" Utility functions for training CIFAR models by adaptive loss scaling. """
import os
import sys
import random
import itertools
import time
import tempfile
import shutil
from contextlib import ExitStack

from PIL import Image
import numpy as np
import cupy as cp
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets import mnist, fashion_mnist
from chainer import training
from chainer.training import extensions

from chainercv.links import PickableSequentialChain
from chainercv import transforms

from chainerlp import notebook_utils
from chainerlp import utils
from chainerlp.hooks import AdaLossMonitor
from chainerlp.links import ResNetCIFAR  # The ResNet model
from chainerlp.links import VGGNetCIFAR
from chainerlp.links import ResNetCIFARv2
from chainerlp import train_utils
from chainerlp.ada_loss import transforms as chainerlp_transforms

from ada_loss.chainer_impl import transforms
from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer_impl.ada_loss import AdaLossChainer
from ada_loss.chainer_impl.ada_loss_recorder import AdaLossRecorder


def train(n_layer,
          init_scale=1,
          scale_upper_bound=128,
          accum_upper_bound=4096,
          method='approx_range',
          update_per_n_iteration=1,
          warmup_attr_ratio=None,
          warmup_n_epoch=None,
          n_class=10,
          manual_seed=0,
          train_batch=128,
          device=-1,
          learnrate=0.1):
    """ Train function """
    utils.set_random_seed(manual_seed, device=device)

    # Recorder for loss scale values
    recorder = AdaLossRecorder(sample_per_n_iter=100)

    with chainer.using_config('dtype', chainer.mixed16):
        if n_layer == 16 or n_layer == 19:
            net_ = VGGNetCIFAR(n_layer, n_class=n_class)
        elif n_layer == 164:
            net_ = ResNetCIFARv2(n_layer, n_class=n_class)
        else:
            net_ = ResNetCIFAR(n_layer, n_class=n_class)

        net = AdaLossScaled(
            net_,
            init_scale=init_scale,
            cfg={
                'loss_scale_method': method,
                'scale_upper_bound': scale_upper_bound,
                'accum_upper_bound': accum_upper_bound,
                'recorder': recorder,
                'update_per_n_iteration': update_per_n_iteration,
                'n_uf_threshold': 1e-3,
            },
            transforms=[
                transforms.AdaLossTransformLinear(),
                transforms.AdaLossTransformConvolution2D(),
                transforms.AdaLossTransformBatchNormalization(),
                # customized transform for chainerlp models
                chainerlp_transforms.AdaLossTransformConv2DBNActiv(),
                chainerlp_transforms.AdaLossTransformBasicBlock(),
                chainerlp_transforms.AdaLossTransformBNActivConv2D(),
                chainerlp_transforms.AdaLossTransformBottleneckv2(),
            ],
            verbose=True)

        hook = AdaLossMonitor(sample_per_n_iter=100,
                              verbose=False,
                              includes=['Grad', 'Deconvolution'])
        utils.set_random_seed(manual_seed, device=device)
        hooks, log = train_utils.train_model_on_cifar(
            net,
            dataset='cifar{}'.format(n_class),
            learnrate=learnrate,
            batchsize=train_batch,
            device=device,
            schedule=[81, 122],
            warmup_attr_ratio=warmup_attr_ratio,
            warmup_n_epoch=warmup_n_epoch,
            hooks=[hook],
            recorder=recorder)

    # post processing
    grad_stats = hooks[0].export_history()
    loss_scale = recorder.export()

    return grad_stats, loss_scale, log


def plot(grad_stats,
         loss_scale,
         log,
         iterations=[0, 100, 1000, 3000],
         grad_name='ReLUGrad2',
         grad_index=0,
         title=None,
         out=None):
    """ Plot the empirical study result on a 3 column figure """
    fig, axes = plt.subplots(ncols=3, figsize=(16, 4))

    # grad stats
    for it in iterations:
        gdf = grad_stats[(grad_stats['iter'] == it)
                         & (grad_stats['label'] == grad_name) &
                         (grad_stats['index'] == grad_index)]
        axes[0].plot(np.arange(1,
                               len(gdf) + 1)[::-1],
                     gdf['nonzero'] / gdf['size'] * 100,
                     label='iter={}'.format(it))
    axes[0].set_xlabel('Layer ID')
    axes[0].set_ylabel('Nonzero (%)')
    axes[0].set_title('Percentage of nonzero activation gradients')
    axes[0].legend()

    # loss scale
    for it in iterations:
        for key in ['unbound', 'final']:
            loss_scale_ = loss_scale[(loss_scale['iter'] == it)
                                     & (loss_scale['key'] == key)]
            axes[1].plot(np.arange(1,
                                   len(loss_scale_) + 1)[::-1],
                         loss_scale_['val'],
                         label='iter={} key={}'.format(it, key))
    axes[1].set_ylabel('Loss scale per layer')
    axes[1].set_xlabel('Layer ID')
    axes[1].set_title('Loss scale per layer')
    axes[1].legend()

    # train log
    axes[2].plot(log['validation/main/accuracy'], label='validation')
    axes[2].plot(log['main/accuracy'], label='train')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_title('Training log')
    axes[2].legend()

    if title is not None:
        plt.suptitle(title)

    return fig