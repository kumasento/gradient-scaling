""" Basic transformations """

import chainer
import chainer.links as L
import chainer.initializers as I

from ada_loss.chainer_impl.transforms.base import AdaLossTransform

# pylint: disable=unused-wildcard-import
from ada_loss.chainer_impl.links import *


class AdaLossTransformLinear(AdaLossTransform):
    """ """

    cls = L.Linear

    def create(self, link, cfg):
        return AdaLossLinear(
            link.in_size,
            out_size=link.out_size,
            nobias=link.b is None,
            ada_loss_cfg=cfg,
        )


class AdaLossTransformConvolution2D(AdaLossTransform):
    """ """

    cls = L.Convolution2D

    def create(self, link, cfg):
        return AdaLossConvolution2D(
            link.in_channels,
            link.out_channels,
            ksize=link.ksize,
            stride=link.stride,
            pad=link.pad,
            dilate=link.dilate,
            groups=link.groups,
            nobias=link.b is None,
            ada_loss_cfg=cfg,
        )


class AdaLossTransformBatchNormalization(AdaLossTransform):
    """ """

    cls = L.BatchNormalization

    def create(self, link, cfg):
        return AdaLossBatchNormalization(link.avg_mean.shape[0], ada_loss_cfg=cfg)
