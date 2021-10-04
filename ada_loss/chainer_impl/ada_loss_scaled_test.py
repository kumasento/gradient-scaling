""" Test the AdaLossScaled wrapper """

import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

from chainer import testing
from chainercv.links import PickableSequentialChain

from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled, loss_scaling
from ada_loss.chainer_impl.links.identity_loss_scaling import IdentityLossScalingWrapper
from ada_loss.chainer_impl.links.ada_loss_linear import AdaLossLinear

np.random.seed(42)


class MLP(PickableSequentialChain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(16, n_units)  # n_in -> n_units
            self.relu1 = lambda x: F.relu(x)  # should be explicit
            self.l2 = L.Linear(n_units, n_units)  # n_units -> n_units
            self.relu2 = lambda x: F.relu(x)  # should be explicit
            self.l3 = L.Linear(n_units, n_out)  # n_units -> n_out


class AdaLossScaledTest(unittest.TestCase):
    """ """

    def test_forward(self):
        """ """
        net1 = MLP(16, 10)
        net2 = AdaLossScaled(net1, init_scale=16.0)

        x = chainer.Variable(np.random.normal(size=(1, 16)).astype("float32"))
        y1 = net1(x)
        y2 = net2(x)
        self.assertTrue(np.allclose(y1.array, y2.array))

    def test_backward(self):
        """ init_scale should be effective """
        net1 = MLP(16, 10)
        net2 = AdaLossScaled(
            net1, init_scale=16.0, transform_functions=False, transforms=None
        )

        x_data = np.random.normal(size=(1, 16)).astype("float32")
        y_data = np.random.normal(size=(1, 10)).astype("float32")

        x1 = chainer.Variable(x_data)
        y1 = net1(x1)
        y1.grad = y_data
        y1.backward()

        x2 = chainer.Variable(x_data)
        y2 = net2(x2)
        y2.grad = y_data
        y2.backward()

        self.assertTrue(np.allclose(x1.grad * 16, x2.grad))

    def test_transform_mlp(self):
        """ Check how transform works """
        np.random.seed(0)

        with chainer.using_config("dtype", "float16"):
            cfg = {
                "loss_scale_method": "fixed",
                "fixed_loss_scale": 2.0,
            }
            net1 = MLP(16, 10)

            x_data = np.random.normal(size=(1, 16)).astype("float16")
            y_data = np.random.normal(size=(1, 10)).astype("float16")

            x = chainer.Variable(x_data)
            y1 = net1(x)
            y1.grad = y_data
            y1.backward()

            x_grad1 = x.grad

            net2 = AdaLossScaled(net1, init_scale=16.0, cfg=cfg, verbose=True)
            x = chainer.Variable(x_data)
            y2 = net2(x)
            self.assertTrue(np.allclose(y1.array, y2.array))

            self.assertTrue(hasattr(net2.link, "l1"))
            self.assertTrue(hasattr(net2.link, "l2"))
            self.assertTrue(hasattr(net2.link, "l3"))
            self.assertIsInstance(getattr(net2.link, "l1"), AdaLossLinear)
            self.assertIsInstance(getattr(net2.link, "l2"), AdaLossLinear)
            self.assertIsInstance(getattr(net2.link, "l3"), AdaLossLinear)

            y2.grad = y_data
            y2.backward()

            self.assertTrue(hasattr(x.grad_var, "loss_scale"))
            # 3 different layers
            self.assertEqual(getattr(x.grad_var, "loss_scale"), 16 * 2 * 2 * 2)

    def test_transform_picked(self):
        """ Test how the transformation performs with picked output """
        np.random.seed(0)

        with chainer.using_config("dtype", "float16"):
            cfg = {
                "loss_scale_method": "fixed",
                "fixed_loss_scale": 2.0,
            }
            net1 = MLP(16, 16)
            net1.pick = ("l1", "l2")

            x_data = np.random.normal(size=(1, 16)).astype("float16")

            x = chainer.Variable(x_data)
            ys1 = net1(x)

            net2 = AdaLossScaled(net1, init_scale=16.0, cfg=cfg, verbose=True)
            x = chainer.Variable(x_data)
            ys2 = net2(x)

            for i in range(len(ys1)):
                self.assertTrue(np.allclose(ys1[i].array, ys2[i].array))

            # try backward propagation
            loss = sum(ys2)
            loss.grad = np.ones_like(loss.array)
            loss.backward()


class IdentityLossScalingTest(unittest.TestCase):
    """ """

    def test_relu(self):
        """ We assume that this loss scale will be propagated to
            the very beginning.  """
        loss_scale = 16.0
        link = chainer.Sequential(
            IdentityLossScalingWrapper(F.relu),
            IdentityLossScalingWrapper(F.relu),
            lambda x: loss_scaling(x, loss_scale),
        )
        x = chainer.Variable(np.random.normal(size=16).astype("float32"))
        y = link(x)
        y.grad = np.random.normal(size=16).astype("float32")
        y.backward()

        # check the propagated back gradient
        self.assertTrue(hasattr(x.grad_var, "loss_scale"))
        self.assertEqual(getattr(x.grad_var, "loss_scale"), loss_scale)

    def test_concat(self):
        """ Test how concatenation works. """
        loss_scale = 16.0
        link = chainer.Sequential(
            IdentityLossScalingWrapper(lambda xs: F.concat(xs, axis=0)),
            # last output should be scaled
            lambda x: loss_scaling(x, loss_scale),
        )

        xs = [
            chainer.Variable(np.random.normal(size=16).astype("float32")),
            chainer.Variable(np.random.normal(size=16).astype("float32")),
        ]
        y = link(xs)
        y.grad = np.random.normal(size=32).astype("float32")
        y.backward()


testing.run_module(__name__, __file__)
