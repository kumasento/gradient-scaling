""" """
import unittest
import numpy as np
import cupy as cp
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing
from chainer.testing import attr

from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet.resblock import Bottleneck
from chainercv.links.model.resnet.resnet import ResNet50

from ada_loss.chainer_impl.functions import loss_scaling
from ada_loss.chainer_impl.transforms.transform_basic import *
from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled

CFG = {
    'loss_scale_method': 'fixed',
    'fixed_loss_scale': 2.0,
    'accum_upper_bound': 1024,
}


class AdaLossTransformLinearTest(unittest.TestCase):
    """ """

    def test_transform(self):
        """ """
        with chainer.using_config('dtype', chainer.mixed16):
            link = L.Linear(10, 20)
            x = chainer.Variable(
                np.random.normal(size=(1, 10)).astype('float16'))
            y1 = link(x)

            link_ = AdaLossTransformLinear()(link, CFG)
            y2 = loss_scaling(link_(x), 16.)
            self.assertTrue(np.allclose(y1.array, y2.array))

            y2.grad = np.ones_like(y2.array, dtype='float16')
            y2.backward()

            self.assertTrue('loss_scale' in x.grad_var.__dict__)
            self.assertEqual(x.grad_var.__dict__['loss_scale'], 16 * 2)


class AdaLossTransformConvolution2DTest(unittest.TestCase):
    """ """

    def test_transform(self):
        """ """
        with chainer.using_config('dtype', chainer.mixed16):
            link = L.Convolution2D(10, 20, ksize=3)
            x = chainer.Variable(
                np.random.normal(size=(1, 10, 4, 4)).astype('float16'))
            y1 = link(x)

            link_ = AdaLossTransformConvolution2D()(link, CFG)
            y2 = loss_scaling(link_(x), 16.)
            self.assertTrue(np.allclose(y1.array, y2.array))

            y2.grad = np.ones_like(y2.array, dtype='float16')
            y2.backward()

            self.assertTrue('loss_scale' in x.grad_var.__dict__)
            self.assertEqual(x.grad_var.__dict__['loss_scale'], 16 * 2)


testing.run_module(__name__, __file__)
