""" Test the correctness of loss_scaling """
import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling
from ada_loss.chainer_impl.functions.ada_loss_branch import AdaLossBranch


class MultiHeadLink(chainer.Chain):
    """ """

    def forward(self, x):
        return AdaLossBranch().apply((x, ))


class LossScalingTest(unittest.TestCase):
    """ Test the performance of loss scaling """

    def test_list_input(self):
        """ Suppose the link to be wrapped takes a list of inputs """
        with chainer.using_config('dtype', chainer.mixed16):
            x = chainer.Variable(
                np.random.normal(size=(1, 2, 3, 3)).astype('float16'))
            link = MultiHeadLink()
            ys = loss_scaling(link(x), 16)

            loss = sum(ys)
            loss.grad = np.ones_like(loss.array)
            loss.backward()

            self.assertTrue('loss_scale' in x.grad_var.__dict__)
            self.assertEquals(x.grad_var.__dict__['loss_scale'], 16)


testing.run_module(__name__, __file__)
