""" Unit test for FixupIdentity """
import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from chainerlp.links import FixupIdentity


class TestFixupIdentity(unittest.TestCase):
    """ Test cases """

    def test_forward(self):
        """ """
        x = chainer.Variable(np.random.random((3, 4, 2, 2)))

        # stride = 2
        fixup_identity = FixupIdentity(2)
        y = fixup_identity(x)
        self.assertTrue((y.data[:, 4:, :, :] == 0).all())

        # stride = 1
        fixup_identity = FixupIdentity(1)
        y = fixup_identity(x)
        self.assertTrue(np.allclose(x.data, y.data))


testing.run_module(__name__, __file__)
