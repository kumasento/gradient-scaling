""" A function node that replicate the input to multiple branches. """

import chainer
from chainer import function_node

from ..ada_loss import AdaLossChainer


class AdaLossBranch(function_node.FunctionNode):
    """ """

    def __init__(self, n_branch=2, ada_loss_cfg=None):
        """ """
        super().__init__()

        self.n_branch = n_branch

        if ada_loss_cfg is None:
            ada_loss_cfg = {}
        self.ada_loss = AdaLossChainer(**ada_loss_cfg)

    def forward(self, inputs):
        """ """
        (x,) = inputs
        return tuple([x] * self.n_branch)

    def backward(self, indexes, grad_outputs):
        """ """
        gs = grad_outputs
        if "loss_scale" not in gs[0].__dict__:
            g = sum(gs)
        else:
            # rescaling only necessary for adaptive
            if self.ada_loss.loss_scale_method == "approx_range":
                gs = self.ada_loss.rescaling(gs)
            g = sum(gs)
            self.ada_loss.set_loss_scale(g, self.ada_loss.grad_loss_scale(gs[0]))
            # print(self.ada_loss.grad_loss_scale(gs[0]))

        return (g,)
