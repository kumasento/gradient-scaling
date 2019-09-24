""" Test the base AdaLoss class. """

import unittest
import numpy as np
import cupy as cp
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing
from chainer.testing import attr

from chainerlp.ada_loss.transforms import *
from chainerlp.links import *
from chainerlp.links.models.resnet import BasicBlock

from ada_loss.chainer.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer.ada_loss_transforms import *


class TransformsTest(unittest.TestCase):
    """ Test ChainerLP custom transforms """

    def test_transform_basic_block(self):
        """ """
        link = BasicBlock(3, 4, stride=2, residual_conv=True)
        tran = AdaLossTransformBasicBlock()
        link_ = tran(link)

        self.assertIsInstance(link_, AdaLossBasicBlock)

        # run inference
        x = chainer.Variable(
            np.random.normal(size=(1, 3, 32, 32)).astype('float32'))
        y1 = link(x)
        y2 = link_(x)
        self.assertTrue(np.allclose(y1.array, y2.array))

    def test_transform_conv2d_bn_activ(self):
        """ """
        link = Conv2DBNActiv(3, 4, ksize=3, stride=1, pad=1)
        tran = AdaLossTransformConv2DBNActiv()
        link_ = tran(link)

        self.assertIsInstance(link_, AdaLossConv2DBNActiv)

        # run inference
        x = chainer.Variable(
            np.random.normal(size=(1, 3, 32, 32)).astype('float32'))
        y1 = link(x)
        y2 = link_(x)
        self.assertTrue(np.allclose(y1.array, y2.array))

    @attr.gpu
    def test_transform_resnet20(self):
        """ """
        cp.random.seed(0)
        cp.cuda.Device(0).use()

        with chainer.using_config('dtype', 'float16'):
            cfg = {
                'loss_scale_method': 'fixed',
                'fixed_loss_scale': 1.,
            }
            net1 = resnet20(n_class=10)
            net1.to_device(0)

            x_data = cp.random.normal(size=(1, 3, 32, 32)).astype('float16')
            x = chainer.Variable(x_data)

            y1 = net1(x)
            net1_params = list(net1.namedparams())

            net2 = AdaLossScaled(net1,
                                 init_scale=1.,
                                 transforms=[
                                     AdaLossTransformLinear(),
                                     AdaLossTransformBasicBlock(),
                                     AdaLossTransformConv2DBNActiv(),
                                 ],
                                 cfg=cfg,
                                 verbose=True)
            net2.to_device(0)
            y2 = net2(x)
            net2_params = list(net2.namedparams())

            self.assertEqual(len(net1_params), len(net2_params))
            for i, p in enumerate(net1_params):
                self.assertTrue(
                    cp.allclose(p[1].array, net2_params[i][1].array))

            self.assertTrue(cp.allclose(y1.array, y2.array))

            # Should not raise error
            y_data = cp.random.normal(size=(1, 10)).astype('float16')
            y2.grad = y_data
            y2.backward()

    @attr.gpu
    def test_transform_resnet18(self):
        """ """
        cp.random.seed(0)
        cp.cuda.Device(0).use()

        with chainer.using_config('dtype', 'float16'):
            cfg = {
                'loss_scale_method': 'fixed',
                'fixed_loss_scale': 1.,
            }
            net1 = resnet18(n_class=10)
            net1.to_device(0)

            x_data = cp.random.normal(size=(2, 3, 224, 224)).astype('float16')
            x = chainer.Variable(x_data)

            y1 = net1(x)
            net1_params = list(net1.namedparams())

            net2 = AdaLossScaled(net1,
                                 init_scale=1.,
                                 transforms=[
                                     AdaLossTransformLinear(),
                                     AdaLossTransformBasicBlock(),
                                     AdaLossTransformConv2DBNActiv(),
                                 ],
                                 cfg=cfg,
                                 verbose=True)
            net2.to_device(0)
            y2 = net2(x)
            net2_params = list(net2.namedparams())

            self.assertEqual(len(net1_params), len(net2_params))
            for i, p in enumerate(net1_params):
                self.assertTrue(
                    cp.allclose(p[1].array, net2_params[i][1].array))

            self.assertTrue(cp.allclose(y1.array, y2.array))

            # Should not raise error
            y_data = cp.random.normal(size=(2, 10)).astype('float16')
            y2.grad = y_data
            y2.backward()

    @attr.gpu
    @attr.slow
    def test_transform_resnet50(self):
        """ """
        cp.random.seed(0)
        cp.cuda.Device(0).use()

        with chainer.using_config('dtype', 'float16'):
            cfg = {
                'loss_scale_method': 'fixed',
                'fixed_loss_scale': 1.,
            }
            net1 = resnet50(n_class=10)
            net1.to_device(0)

            x_data = cp.random.normal(size=(2, 3, 224, 224)).astype('float16')
            x = chainer.Variable(x_data)

            y1 = net1(x)
            net1_params = list(net1.namedparams())

            net2 = AdaLossScaled(net1,
                                 init_scale=1.,
                                 transforms=[
                                     AdaLossTransformLinear(),
                                     AdaLossTransformBottleneck(),
                                     AdaLossTransformConv2DBNActiv(),
                                 ],
                                 cfg=cfg,
                                 verbose=True)
            net2.to_device(0)
            y2 = net2(x)
            net2_params = list(net2.namedparams())

            self.assertEqual(len(net1_params), len(net2_params))
            for i, p in enumerate(net1_params):
                self.assertTrue(
                    cp.allclose(p[1].array, net2_params[i][1].array))

            self.assertTrue(cp.allclose(y1.array, y2.array))

            # Should not raise error
            y_data = cp.random.normal(size=(2, 10)).astype('float16')
            y2.grad = y_data
            y2.backward()


testing.run_module(__name__, __file__)