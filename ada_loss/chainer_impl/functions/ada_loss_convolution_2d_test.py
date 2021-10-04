""" Unit test for AdaLossConvolution2D """

import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from ada_loss.chainer_impl.ada_loss_chainer import AdaLossChainer
from ada_loss.chainer_impl.functions.ada_loss_convolution_2d import (
    ada_loss_convolution_2d,
)
from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling
from chainer import testing

np.random.seed(0)


class AdaLossConvolution2DTest(unittest.TestCase):
    def test_backward(self):
        """ """
        x = chainer.Variable(np.random.normal(size=(1, 3, 4, 4)).astype("float16"))
        W = chainer.Variable(np.random.normal(size=(4, 3, 3, 3)).astype("float16"))
        y = loss_scaling(
            ada_loss_convolution_2d(
                x, W, ada_loss=AdaLossChainer(loss_scale_method="fixed")
            ),
            2.0,
        )
        y.grad = np.ones_like(y.array)
        y.backward()

        self.assertTrue(hasattr(x.grad_var, "loss_scale"))
        self.assertTrue(hasattr(W.grad_var, "loss_scale"))
        # scaled down
        self.assertEqual(getattr(W.grad_var, "loss_scale"), 1.0)


testing.run_module(__name__, __file__)
