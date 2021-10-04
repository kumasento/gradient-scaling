""" Test the base AdaLoss class. """

import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.ada_loss import AdaLoss

np.random.seed(0)


class AdaLossTest(unittest.TestCase):
    """ Test the AdaLoss base class """

    def _test_power_of_two(self, src, dst, dtype):
        """ wrapper of the tests """
        ada_loss = AdaLoss()
        self.assertEqual(dst, ada_loss.get_power_of_two_scale(src))

    def test_power_of_two(self):
        """ The power of 2 scale rounding method """
        self._test_power_of_two(1.0, 1.0, np.float16)
        self._test_power_of_two(1.23, 1.0, np.float16)
        self._test_power_of_two(0.23, 0.125, np.float16)


testing.run_module(__name__, __file__)
