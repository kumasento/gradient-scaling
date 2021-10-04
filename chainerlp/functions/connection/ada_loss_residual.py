""" Residual split with support for adaptive loss scaling """

import chainer


class AdaLossResidual(chainer.function.function_node.FunctionNode):
    """ """

    def __init__(self):
        """ CTOR """
        super().__init__()

    def forward(self, inputs):
        """ """
        (x,) = inputs
        return x, x

    def backward(self, input_indexes, grad_outputs):
        """ """
