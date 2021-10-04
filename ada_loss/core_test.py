""" Test the base AdaLoss class. """

import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import testing

from ada_loss.core import AdaLoss

np.random.seed(0)


class AdaLossTest(unittest.TestCase):
    """ Test the AdaLoss base class """

    def _test_power_of_two(self, src, dst, dtype):
        """ wrapper of the tests """
        ada_loss_ = AdaLoss()
        self.assertEqual(dst, ada_loss_.get_power_of_two_scale(src))

    def test_power_of_two(self):
        """ The power of 2 scale rounding method """
        self._test_power_of_two(1.0, 1.0, np.float16)
        self._test_power_of_two(1.23, 1.0, np.float16)
        self._test_power_of_two(0.23, 0.125, np.float16)


testing.run_module(__name__, __file__)
