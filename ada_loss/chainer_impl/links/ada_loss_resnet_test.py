import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.chainer_impl.links.ada_loss_resnet import AdaLossBasicBlock, AdaLossBottleneckBlock
from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling


class AdaLossBasicBlockTest(unittest.TestCase):
    """ """

    def test_backward(self):
        with chainer.using_config('dtype', chainer.mixed16):
            x = chainer.Variable(
                np.random.normal(size=(1, 3, 4, 4)).astype('float16'))
            link = chainer.Sequential(
                AdaLossBasicBlock(3,
                                  3,
                                  ada_loss_cfg={
                                      'fixed_loss_scale': 2,
                                      'loss_scale_method': 'fixed'
                                  }),
                lambda x: loss_scaling(x, 16.),
            )
            y = link(x)
            y.grad = np.ones_like(y.array, dtype=np.float16)
            y.backward()

        # grad_var can propagate
        self.assertTrue(hasattr(x.grad_var, 'loss_scale'))
        # NOTE: left term is the residual branch
        self.assertEqual(getattr(x.grad_var, 'loss_scale'), 16 * 2 * 2)


class AdaLossBottleneckBlockTest(unittest.TestCase):
    """ Test the bottleneck block. """

    def test_backward(self):
        """ Backward function """
        with chainer.using_config('dtype', chainer.mixed16):
            x = chainer.Variable(
                np.random.normal(size=(1, 3, 4, 4)).astype('float16'))
            link = chainer.Sequential(
                AdaLossBottleneckBlock(3,
                                       2,
                                       3,
                                       ada_loss_cfg={
                                           'fixed_loss_scale': 2,
                                           'loss_scale_method': 'fixed'
                                       }),
                lambda x: loss_scaling(x, 16.),
            )
            y = link(x)
            y.grad = np.ones_like(y.array, dtype=np.float16)
            y.backward()

        # grad_var can propagate
        self.assertTrue(hasattr(x.grad_var, 'loss_scale'))
        # NOTE: left term is the residual branch
        self.assertEqual(getattr(x.grad_var, 'loss_scale'), 16 * 2 * 2 * 2)


testing.run_module(__name__, __file__)
