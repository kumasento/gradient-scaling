""" Linear function that supports adaptive loss scaling. """

import math
import numpy as np
import functools

import chainer
import chainer.functions as F
from chainer import utils
# to be inherited
from chainer.functions.connection import linear

from chainerlp.ada_loss.ada_loss_chainer import AdaLossChainer


class AdaLossLinearFunction(linear.LinearFunction):
    """ The FunctionNode for performing ada_loss_linear computation. """

    def __init__(self, **kwargs):
        """ CTOR.
            We need to use id and scale_map to retrieve the factors
            multiplied by nodes preceded to the current one. 
        """
        super().__init__()

        self.ada_loss = AdaLossChainer(**kwargs)

    def backward(self, indexes, grad_outputs):
        """ The gradient for the output will be scaled """
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        s_gy, u_gy = self.ada_loss.loss_scaling(gy, W)

        # Actual gradient calculation
        ret = []
        with chainer.using_config('use_ideep', self._config_use_ideep):
            if 0 in indexes:
                gx, = linear.LinearGradData().apply((W, s_gy))
                ret.append(F.cast(gx, x.dtype))
            if 1 in indexes:
                gW, = linear.LinearGradWeight(W.dtype).apply((x, u_gy))
                ret.append(F.cast(gW, W.dtype))
            if 2 in indexes:
                gb = chainer.functions.sum(u_gy, axis=0)
                ret.append(gb)

        return ret


def ada_loss_linear(x, W, b=None, n_batch_axes=1, **kwargs):
    """ Simply replace the LinearFunction in linear to AdaLossLinear """
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
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

    y, = AdaLossLinearFunction(**kwargs).apply(args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1, ))
    return y