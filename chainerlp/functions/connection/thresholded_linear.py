""" LinearFunction with threshold """
import numpy

from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
import chainer.functions
from chainer.graph_optimizations import static_code
from chainer import utils
from chainer.utils import type_check
import chainerx

from chainer.functions.connection.linear import LinearFunction, LinearGradData, LinearGradWeight

from chainerlp.functions.math.threshold import Threshold


class ThresholdedLinearFunction(LinearFunction):
    """ Thresholding elementwise multiplication before summing """

    def __init__(self, threshold=6e-8):
        super(ThresholdedLinearFunction, self).__init__()

        self.threshold = threshold
        self.threshold_func = Threshold(threshold)

    def forward(self, inputs):
        """ Customized forward function """
        if len(inputs) == 3:
            x, W, b = inputs
        else:
            (x, W), b = inputs, None

        if (isinstance(x, numpy.ndarray)
                and not (x.flags.c_contiguous or x.flags.f_contiguous)
                and 1 in x.shape):
            x = numpy.ascontiguousarray(x)

        xp = cuda.get_array_module(x)
        N, K = x.shape
        M = W.shape[0]

        x_ = xp.repeat(x.reshape((N, 1, K)), M, axis=1)
        W_ = xp.repeat(W.reshape((1, M, K)), N, axis=0)

        # NOTE: we re-use the threshold node
        # Here might have memory issue since the multiplication result
        # of shape (N, M, K) is retained.
        y_ = self.threshold_func.apply((xp.multiply(x_, W_), ))[0]
        y = xp.sum(y_.array, axis=2)

        if len(inputs) == 3:
            self.static_add_bias(inputs=[b], outputs=[y])

        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        N, K = x.shape
        M = W.shape[0]

        gy, = grad_outputs
        ret = []

        # duplicate gy into k copies
        xp = cuda.get_array_module(x)
        gy_ = xp.repeat(gy.reshape((N, M, 1)).array, K, axis=2)

        # calculate the gradient after passing through the threshold function
        # NOTE: indexes are not useful here
        ggy_, = self.threshold_func.backward(indexes, (gy_, ))

        if 0 in indexes:
            gx, = _LinearGradData().apply((W, ggy_))
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            gW, = _LinearGradWeight(W.dtype).apply((x, ggy_))
            ret.append(chainer.functions.cast(gW, W.dtype))
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)

        return ret


class _LinearGradWeight(LinearGradWeight):
    """ """

    def forward(self, inputs):
        """ """
        self.retain_inputs((0, 1))
        x, gy = inputs  # gy has shape N x M x K, which is also ggy
        xp = cuda.get_array_module(x)

        if (isinstance(gy, numpy.ndarray)
                and not (gy.flags.c_contiguous or gy.flags.f_contiguous)
                and 1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gW = xp.sum(xp.multiply(gy, x.reshape((x.shape[0], 1, x.shape[1]))),
                    axis=0).astype(self._w_dtype, copy=False)
        return gW,

    def backward(self, indexes, grad_outputs):
        """ """
        raise NotImplementedError()


class _LinearGradData(LinearGradData):
    """ """

    def forward(self, inputs):
        """ """
        # Generic implementation
        self.retain_inputs((0, 1))
        W, gy = inputs
        xp = cuda.get_array_module(W)

        if (isinstance(gy, numpy.ndarray)
                and not (gy.flags.c_contiguous or gy.flags.f_contiguous)
                and 1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gx = xp.sum(xp.multiply(gy, W), axis=1).astype(gy.dtype, copy=False)
        return gx,

    def backward(self, indexes, grad_outputs):
        """ """
        raise NotImplementedError()


def thresholded_linear(x, W, b=None, n_batch_axes=1, threshold=6e-8):
    """ """
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

    y, = ThresholdedLinearFunction(threshold=threshold).apply(args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1, ))
    return y