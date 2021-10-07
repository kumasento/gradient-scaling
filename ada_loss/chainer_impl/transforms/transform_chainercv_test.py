""" """
import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp
import numpy as np
from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer_impl.ada_loss_transforms import AdaLossTransformLinear
from ada_loss.chainer_impl.functions import loss_scaling
from ada_loss.chainer_impl.transforms.transform_chainercv import *
from chainer import testing
from chainer.testing import attr
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet.resblock import Bottleneck
from chainercv.links.model.resnet.resnet import ResNet50

CFG = {
    "loss_scale_method": "fixed",
    "fixed_loss_scale": 2.0,
    "accum_upper_bound": 1024,
}


class AdaLossTransformConv2DBNActivTest(unittest.TestCase):
    """ """

    def test_transform(self):
        """ """
        with chainer.using_config("dtype", chainer.mixed16):
            link = Conv2DBNActiv(3, 4, ksize=3, stride=2, pad=1)
            x = chainer.Variable(np.random.normal(size=(1, 3, 4, 4)).astype("float16"))
            y1 = link(x)

            link_ = AdaLossTransformConv2DBNActiv()(link, CFG)
            y2 = loss_scaling(link_(x), 16.0)
            self.assertTrue(np.allclose(y1.array, y2.array))

            y2.grad = np.ones_like(y2.array, dtype="float16")
            y2.backward()

            self.assertTrue("loss_scale" in x.grad_var.__dict__)
            self.assertEqual(x.grad_var.__dict__["loss_scale"], 16 * 2)


class AdaLossTransformBottleneckTest(unittest.TestCase):
    """ """

    def test_transform(self):
        with chainer.using_config("dtype", chainer.mixed16):
            # the original link
            # TODO: stride first assertion
            link = Bottleneck(4, 2, 4, stride=2, residual_conv=True, stride_first=True)

            x = chainer.Variable(np.random.normal(size=(1, 4, 8, 8)).astype("float16"))
            y1 = link(x)

            link_ = AdaLossTransformBottleneck()(link, CFG)
            y2 = loss_scaling(link_(x), 16.0)
            self.assertTrue(np.allclose(y1.array, y2.array))

            y2.grad = np.ones_like(y2.array, dtype="float16")
            y2.backward()

            self.assertTrue("loss_scale" in x.grad_var.__dict__)
            self.assertEqual(x.grad_var.__dict__["loss_scale"], 16 * 2 * 2 * 2)


class ResNetTransformTest(unittest.TestCase):
    """ Try to transform a standard ChainerCV ResNet model """

    def test_transform_resnet50(self):
        """ """
        if not cp.cuda.is_available():
            return
        cp.random.seed(0)
        cp.cuda.Device(0).use()

        with chainer.using_config("dtype", chainer.mixed16):
            net = ResNet50(arch="he")  # stride_first is True
            net.to_gpu()

            x = chainer.Variable(
                cp.random.normal(size=(1, 3, 224, 224)).astype("float16")
            )
            y1 = net(x)

            net_ = AdaLossScaled(
                net,
                init_scale=16.0,
                transforms=[
                    AdaLossTransformLinear(),
                    AdaLossTransformBottleneck(),
                    AdaLossTransformConv2DBNActiv(),
                ],
                verbose=False,
                cfg=CFG,
            )
            net_.to_gpu()
            y2 = net_(x)
            self.assertTrue(cp.allclose(y1.array, y2.array))

            y2.grad = cp.ones_like(y2.array, dtype="float16")
            y2.backward()

            self.assertTrue("loss_scale" in x.grad_var.__dict__)
            self.assertEqual(
                x.grad_var.__dict__["loss_scale"], CFG["accum_upper_bound"]
            )


testing.run_module(__name__, __file__)
