""" Test """

import unittest
import numpy as np
import cupy as cp
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing
from chainer.testing import attr

from models.ssd.ssd_vgg16 import VGG16, VGG16Extractor300, VGG16Extractor512
from chainercv.links.model.ssd import ssd_vgg16


class VGG16Test(unittest.TestCase):
    def test_forward(self):
        """ """
        cp.cuda.Device(0).use()

        vgg16a = VGG16()
        vgg16a.to_gpu()
        vgg16b = ssd_vgg16.VGG16()
        vgg16b.to_gpu()

        x = chainer.Variable(cp.random.normal(size=(1, 3, 224, 224)).astype("float32"))
        vgg16a(x)  # initialization
        vgg16b(x)
        vgg16a.copyparams(vgg16b)

        y1 = vgg16a(x)
        # The original VGG16 outputs a norm branch, ignored here
        ys2 = vgg16b(x)
        self.assertTrue(cp.allclose(y1.array, ys2[1].array))


class VGG16ExtractorTest(unittest.TestCase):
    """ norm4 is removed """

    def test_forward_300(self):
        """ """
        cp.cuda.Device(0).use()

        vgg16a = VGG16Extractor300()
        vgg16a.to_gpu()
        vgg16b = ssd_vgg16.VGG16Extractor300()
        vgg16b.to_gpu()

        x = chainer.Variable(cp.random.normal(size=(1, 3, 800, 800)).astype("float32"))
        vgg16a(x)  # initialization
        vgg16b(x)
        vgg16a.copyparams(vgg16b)

        ys1 = vgg16a(x)
        # The original VGG16 outputs a norm branch, ignored here
        ys2 = vgg16b(x)
        for i in range(1, len(ys1)):
            self.assertTrue(cp.allclose(ys1[i].array, ys2[i].array))

    def test_forward_500(self):
        """ """
        cp.cuda.Device(0).use()

        vgg16a = VGG16Extractor512()
        vgg16a.to_gpu()
        vgg16b = ssd_vgg16.VGG16Extractor512()
        vgg16b.to_gpu()

        x = chainer.Variable(cp.random.normal(size=(1, 3, 800, 800)).astype("float32"))
        vgg16a(x)  # initialization
        vgg16b(x)
        vgg16a.copyparams(vgg16b)

        ys1 = vgg16a(x)
        # The original VGG16 outputs a norm branch, ignored here
        ys2 = vgg16b(x)
        for i in range(1, len(ys1)):
            self.assertTrue(cp.allclose(ys1[i].array, ys2[i].array))


testing.run_module(__name__, __file__)
