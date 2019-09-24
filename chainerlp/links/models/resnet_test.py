""" Test implemented ResNet utilities """

import unittest

import numpy as np

import chainer
from chainer import testing

from chainerlp import utils
from chainerlp.links import *

from chainercv.links.model import resnet


class ResNetTest(unittest.TestCase):

    _model_sizes = {
        20: 0.27,
        32: 0.46,
        44: 0.66,
        56: 0.85,
        110: 1.7,
        1202: 19.3,  # NOTE: tweaked
        18: 11.7,
        50: 25.6,
        101: 44.5,
        152: 60.2,
    }

    def test_basic_block(self):
        """ Try to initialize BasicBlock and run through it. """
        block = BasicBlock(32, 64, residual_conv=True)
        data = np.random.random((4, 32, 8, 8)).astype(np.float32)
        x = chainer.Variable(data)
        y = block(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (4, 64, 8, 8))

    def test_res_basic_block(self):
        """ Test the ResBasicBlock. """
        res_block = ResBasicBlock(3, 32, 64, 1)

        data = np.random.random((4, 32, 8, 8)).astype(np.float32)
        x = chainer.Variable(data)
        y = res_block(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (4, 64, 8, 8))

    def test_res_bottleneck_block(self):
        """ Test the ResBottleneckBlock. """
        res_block = ResBottleneckBlock(3, 64, 32, 256, 1)

        data = np.random.random((4, 64, 8, 8)).astype(np.float32)
        x = chainer.Variable(data)
        y = res_block(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (4, 256, 8, 8))

    def test_fixup_resnet(self):
        net = ResNetCIFAR(20, n_class=10, use_fixup=True)

    def _test_resnet(self, n_layer, skip_infer=True):
        """ Test the base class of ResNet: """
        if n_layer in [18, 50, 101, 152]:
            n_class = 1000
            img_size = 224
            net = ResNet(n_layer, n_class=n_class)
            self.assertAlmostEqual(utils.get_model_size(net) / 1e6,
                                   self._model_sizes[n_layer],
                                   places=1)
        else:
            n_class = 10
            img_size = 32
            net = ResNetCIFAR(n_layer, n_class=n_class)
            self.assertAlmostEqual(
                utils.get_model_size(net, excludes=['gamma', 'beta']) / 1e6,
                self._model_sizes[n_layer],
                places=1)

        if not skip_infer:
            batch_size = 2
            data = np.random.random(
                (batch_size, 3, img_size, img_size)).astype(np.float32)
            x = chainer.Variable(data)
            y = net(x)  # NOTE: should not raise error
            self.assertEqual(y.shape, (batch_size, n_class))

    def test_resnet(self):
        for n_layer in [20, 32, 44, 56, 110, 1202, 18, 50, 101, 152]:
            self._test_resnet(n_layer)

    def test_first_bn_mixed16(self):
        net = ResNet(18, n_class=1000, first_bn_mixed16=True)
        self.assertEqual(net.conv1.bn.avg_mean.dtype, 'float32')


testing.run_module(__name__, __file__)
