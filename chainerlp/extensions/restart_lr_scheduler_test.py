""" Test the behaviour of the RestartLRScheduler. """

import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from chainerlp.extensions import RestartLRScheduler


class RestartLRSchedulerTest(unittest.TestCase):
    """ """

    def test_train(self):
        pass


testing.run_module(__name__, __file__)
