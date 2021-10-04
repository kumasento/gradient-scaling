""" The threshold function to mimic the underflow effect. """

import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer import backend
from chainer.utils import type_check


class Threshold(function_node.FunctionNode):
    """ Underflow threshold function """

    def __init__(self, x_min):
        """ if abs(x) <= x_min returns 0 """
        self.x_min = x_min

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ("x",))
        (x_type,) = in_types
        type_check.expect(x_type.dtype.kind == "f")

    def forward_cpu(self, x):
        self.retain_inputs((0,))  # keep input data
        # TODO: might be able to optimize
        return (utils.force_array((numpy.abs(x[0]) >= self.x_min) * x[0]),)

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        # TODO: might be able to optimize
        return ((cuda.cupy.abs(x[0]) >= self.x_min) * x[0],)

    def backward(self, indexes, grad_outputs):
        (x,) = self.get_retained_inputs()
        return ThresholdGrad(x.data, self.x_min).apply(grad_outputs)


class ThresholdGrad(function_node.FunctionNode):
    def __init__(self, x, x_min):
        xp = backend.get_array_module(x)
        self.cond = xp.abs(x) >= x_min

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ("gy",))
        type_check.expect(in_types[0].dtype.kind == "f")

    def forward_cpu(self, inputs):
        return (utils.force_array(inputs[0] * self.cond),)

    def forward_gpu(self, inputs):
        gx = cuda.elementwise(
            "T gy, bool cond", "T gx", "gx = cond ? gy : T(0)", "threshold_bwd"
        )(inputs[0], self.cond)
        return (gx,)

    def backward(self, indexes, grad_outputs):
        return (grad_outputs[0] * self.cond,)


def threshold(x, x_min):
    return Threshold(x_min).apply((x,))[0]
