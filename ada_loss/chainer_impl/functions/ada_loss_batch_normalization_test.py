"""Unit test for AdaLossBatchNormalization """

import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling
from ada_loss.chainer_impl.functions.ada_loss_batch_normalization import ada_loss_batch_normalization

np.random.seed(0)


class AdaLossBatchNormalizationTest(unittest.TestCase):
    def test_backward(self):
        """ """
        x = chainer.Variable(
            np.random.normal(size=(1, 3, 4, 4)).astype('float16'))
        gamma = chainer.Variable(np.random.normal(size=(3)).astype('float16'))
        beta = chainer.Variable(np.random.normal(size=(3)).astype('float16'))

        y = loss_scaling(ada_loss_batch_normalization(x, gamma, beta), 2.)
        y.grad = np.ones_like(y.array)
        y.backward()

        self.assertTrue(hasattr(x.grad_var, 'loss_scale'))
        self.assertEqual(getattr(x.grad_var, 'loss_scale'), 2.0)
        self.assertTrue(hasattr(gamma.grad_var, 'loss_scale'))
        self.assertEqual(getattr(gamma.grad_var, 'loss_scale'), 1.0)
        self.assertTrue(hasattr(beta.grad_var, 'loss_scale'))
        self.assertEqual(getattr(beta.grad_var, 'loss_scale'), 1.0)


testing.run_module(__name__, __file__)
