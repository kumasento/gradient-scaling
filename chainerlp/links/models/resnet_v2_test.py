""" Test resnet_v2 """

import unittest

import numpy as np

import chainer
from chainer import testing

from chainerlp import utils
from chainerlp.links import *

from chainercv.links.model import resnet


class ResNetTest(unittest.TestCase):

    _model_sizes = {
        164: 1.7,
        1001: 10.3,
    }

    def _test_resnet(self, n_layer):
        """ Test the base class of ResNet: """
        n_class = 10
        img_size = 32
        net = ResNetCIFARv2(n_layer, n_class=n_class)
        self.assertAlmostEqual(utils.get_model_size(net) / 1e6,
                               self._model_sizes[n_layer], places=1)

        batch_size = 2
        data = np.random.random(
            (batch_size, 3, img_size, img_size)).astype(np.float32)
        x = chainer.Variable(data)
        y = net(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (batch_size, n_class))

    def test_resnet(self):
        for n_layer in [164, 1001]:
            self._test_resnet(n_layer)


testing.run_module(__name__, __file__)
