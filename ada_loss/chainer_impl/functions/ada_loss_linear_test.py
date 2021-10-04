import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from ada_loss.chainer_impl.ada_loss_chainer import AdaLossChainer
from ada_loss.chainer_impl.functions.ada_loss_linear import ada_loss_linear
from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling
from chainer import testing

np.random.seed(0)


class AdaLossLinearTest(unittest.TestCase):
    """ """

    def test_forward(self):
        dtype = np.float16
        x_data = np.random.normal(size=(2, 4)).astype(dtype)
        W_data = np.random.normal(size=(3, 4)).astype(dtype)
        b_data = np.random.normal(size=(3)).astype(dtype)

        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)
        y1 = F.linear(x, W, b=b)
        y2 = ada_loss_linear(x, W, b=b, ada_loss=AdaLossChainer())
        self.assertTrue(np.allclose(y1.array, y2.array))

    def test_backward(self):
        dtype = np.float16
        x_data = np.random.normal(size=(2, 4)).astype(dtype)
        W_data = np.random.normal(size=(3, 4)).astype(dtype)
        b_data = np.random.normal(size=(3)).astype(dtype)
        g_data = np.random.normal(size=(2, 3)).astype(dtype)

        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)

        # no loss scaling
        y1 = F.linear(x, W, b=b)
        y1.grad = g_data
        y1.backward()

        W_grad1 = W.grad
        x_grad1 = x.grad
        b_grad1 = b.grad

        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)
        # with loss scaling
        y2 = loss_scaling(
            ada_loss_linear(
                x,
                W,
                b=b,
                ada_loss=AdaLossChainer(
                    loss_scale_method="fixed", fixed_loss_scale=2.0
                ),
            ),
            2.0,
        )
        y2.grad = g_data
        y2.backward()

        self.assertTrue(np.allclose(x.grad, x_grad1 * 4))
        self.assertTrue(np.allclose(W.grad, W_grad1))
        self.assertTrue(np.allclose(b.grad, b_grad1))


testing.run_module(__name__, __file__)
