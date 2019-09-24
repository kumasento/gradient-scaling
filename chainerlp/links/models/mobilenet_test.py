""" Test the MobileNet arch. """

import unittest

import numpy as np

import chainer
from chainer import testing

from chainerlp import utils
from chainerlp.links import MobileNetV1


class MobileNetTest(unittest.TestCase):
    """ Test VGG """

    def _test_net(self):
        net = MobileNetV1(n_class=10)

        data = np.random.random((1, 3, 32, 32)).astype(np.float32)
        x = chainer.Variable(data)
        y = net(x)  # NOTE: should not raise error
        self.assertEqual(y.shape, (1, 10))

    def test_net(self):
        self._test_net()


testing.run_module(__name__, __file__)
