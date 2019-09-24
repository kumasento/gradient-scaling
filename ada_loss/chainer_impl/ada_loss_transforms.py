""" Implement the transformations we need to use
    to convert a link to an adaptive loss scaled link. """
# NOTE: this file is deprecated

import chainer
import chainer.links as L
import chainer.initializers as I

# pylint: disable=unused-wildcard-import
from ada_loss.chainer_impl.links import *

__all__ = [
    'AdaLossTransformLinear',
    'AdaLossTransformConvolution2D',
]


class AdaLossTransform(object):
    """ The base class """

    def __call__(self, link, cfg):
        """ Entry """
        raise NotImplementedError(
            'This call function should be implemented properly')


class AdaLossTransformLinear(AdaLossTransform):
    """ """
    cls = L.Linear

    def __call__(self, link, cfg, initialW=I.HeNormal()):
        assert isinstance(link, self.cls)
        link_ = AdaLossLinear(link.in_size,
                              out_size=link.out_size,
                              nobias=link.b is None,
                              ada_loss_cfg=cfg)
        link_.copyparams(link)
        return link_



class AdaLossTransformConvolution2D(AdaLossTransform):
    """ """
    cls = L.Convolution2D

    def __call__(self, link, cfg, initialW=I.HeNormal()):
        assert isinstance(link, self.cls)
        link_ = AdaLossConvolution2D(link.in_channels,
                                     link.out_channels,
                                     ksize=link.ksize,
                                     stride=link.stride,
                                     pad=link.pad,
                                     dilate=link.dilate,
                                     groups=link.groups,
                                     nobias=link.b is None,
                                     ada_loss_cfg=cfg)
        link_.copyparams(link)
        return link_