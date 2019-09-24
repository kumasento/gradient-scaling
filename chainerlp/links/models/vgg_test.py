""" Test the VGG arch. """

import unittest

import numpy as np

import chainer
from chainer import testing

from chainerlp import utils
from chainerlp.links import VGGNetCIFAR


class VGGTest(unittest.TestCase):
    """ Test VGG """

    def _test_vgg_net(self, n_layer, use_batchnorm=False):
        net = VGGNetCIFAR(n_layer, n_class=10, use_batchnorm=use_batchnorm)

        data = np.random.random((1, 3, 32, 32)).astype(np.float32)
        x = chainer.Variable(data)
        y = net(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (1, 10))

    def test_vgg(self):
        self._test_vgg_net(11)
        self._test_vgg_net(13)
        self._test_vgg_net(16)
        self._test_vgg_net(19)
        self._test_vgg_net(11, use_batchnorm=True)
        self._test_vgg_net(13, use_batchnorm=True)
        self._test_vgg_net(16, use_batchnorm=True)
        self._test_vgg_net(19, use_batchnorm=True)


testing.run_module(__name__, __file__)
