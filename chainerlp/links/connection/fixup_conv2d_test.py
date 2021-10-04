""" Unit test for FixupConv2D """
import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from chainerlp.links import FixupConv2D


class TestFixupConv2D(unittest.TestCase):
    """ Test cases """

    def test_forward(self):
        """ Forward should work correctly """
        fixup_conv2d = FixupConv2D(32, 32, ksize=3, pad=1)
        x = chainer.Variable(np.random.random((1, 32, 4, 4)).astype("float32"))
        y = fixup_conv2d(x)
        z = F.relu(F.convolution_2d(x, fixup_conv2d.conv.W, pad=1))
        self.assertTrue(np.allclose(y.array, z.array))

        # setup bias_in
        fixup_conv2d.bias_in.data = np.array([0.1], dtype=np.float32)
        y = fixup_conv2d(x)
        z = F.relu(F.convolution_2d(x + 0.1, fixup_conv2d.conv.W, pad=1))
        self.assertTrue(np.allclose(y.array, z.array))

        # bias_out
        fixup_conv2d.bias_out.data = np.array([0.2], dtype=np.float32)
        y = fixup_conv2d(x)
        z = F.relu(F.convolution_2d(x + 0.1, fixup_conv2d.conv.W, pad=1) + 0.2)
        self.assertTrue(np.allclose(y.array, z.array))

        # scale
        fixup_conv2d.scale.data = np.array([0.1], dtype=np.float32)
        y = fixup_conv2d(x)
        z = F.relu(F.convolution_2d(x + 0.1, fixup_conv2d.conv.W, pad=1) * 0.1 + 0.2)
        self.assertTrue(np.allclose(y.array, z.array))

    def test_forward_wo_scale(self):
        """ Initialize FixupConv2D without scale """
        fixup_conv2d = FixupConv2D(32, 32, ksize=3, pad=1, use_scale=False)
        self.assertIsNone(fixup_conv2d.scale)
        x = chainer.Variable(np.random.random((1, 32, 4, 4)).astype("float32"))
        y = fixup_conv2d(x)
        z = F.relu(F.convolution_2d(x, fixup_conv2d.conv.W, pad=1))
        self.assertTrue(np.allclose(y.array, z.array))

    def test_backward(self):
        """ Should update bias and scale in the backward computation. """

        fixup_conv2d = FixupConv2D(3, 3, ksize=3, pad=1, activ=None)
        grad = np.random.random((1, 3, 4, 4)).astype("float32")
        x = chainer.Variable(np.random.random((1, 3, 4, 4)).astype("float32"))

        y = fixup_conv2d(x)
        y.grad = grad
        y.backward()


testing.run_module(__name__, __file__)
