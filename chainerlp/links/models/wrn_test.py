""" Test Wide-ResNet. """

import unittest

import numpy as np

import chainer
from chainer import testing

from chainerlp import utils
from chainerlp.links import *


class WideResNetTest(unittest.TestCase):

    def _test_wrn(self, model, expected_size):
        """ Test the base class of ResNet: """
        n_class = 10
        img_size = 32
        net = model(n_class=n_class)
        self.assertAlmostEqual(utils.get_model_size(
            net) / 1e6, expected_size, places=1)

        batch_size = 2
        data = np.random.random(
            (batch_size, 3, img_size, img_size)).astype(np.float32)
        x = chainer.Variable(data)
        y = net(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (batch_size, n_class))

    def test_wrn(self):
        self._test_wrn(wrn_28_10, 36.5)
        self._test_wrn(wrn_40_4, 8.9)
        self._test_wrn(wrn_16_8, 11)


testing.run_module(__name__, __file__)
