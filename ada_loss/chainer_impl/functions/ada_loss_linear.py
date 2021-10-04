""" Linear function that supports adaptive loss scaling. """

import functools
import math

import chainer
import chainer.functions as F
import numpy as np
from ada_loss.chainer_impl.ada_loss_chainer import AdaLossChainer
from chainer import utils

# to be inherited
from chainer.functions.connection import linear


class AdaLossLinearFunction(linear.LinearFunction):
    """ The FunctionNode for performing ada_loss_linear computation. """

    def __init__(self, ada_loss=None):
        """ CTOR.
            We need to use id and scale_map to retrieve the factors
            multiplied by nodes preceded to the current one. 
        """
        super().__init__()

        self.ada_loss = ada_loss
        self.ada_loss.func = self

    def backward(self, indexes, grad_outputs):
        """ The gradient for the output will be scaled """
        x, W = self.get_retained_inputs()
        (gy,) = grad_outputs
        gy_, prev_scale = self.ada_loss.loss_scaling(gy, W)

        ret = []
        with chainer.using_config("use_ideep", self._config_use_ideep):
            if 0 in indexes:
                (gx,) = linear.LinearGradData().apply((W, gy_))
                self.ada_loss.set_loss_scale(gx, self.ada_loss.grad_loss_scale(gy_))
                ret.append(F.cast(gx, x.dtype))
            if 1 in indexes:
                (gW,) = linear.LinearGradWeight(W.dtype).apply((x, gy))
                gW_ = self.ada_loss.get_unscaled_gradient(gW, prev_scale)
                ret.append(F.cast(gW_, W.dtype))
            if 2 in indexes:
                gb = chainer.functions.sum(gy, axis=0)
                gb_ = self.ada_loss.get_unscaled_gradient(gb, prev_scale)
                ret.append(gb_)

        return ret


def ada_loss_linear(x, W, b=None, n_batch_axes=1, ada_loss=None):
    """ Simply replace the LinearFunction in linear to AdaLossLinear """
    if n_batch_axes <= 0:
        raise ValueError("n_batch_axes should be greater than 0.")
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = utils.size_of_shape(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if b is None:
        args = x, W
    else:
        args = x, W, b

    (y,) = AdaLossLinearFunction(ada_loss=ada_loss).apply(args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y
