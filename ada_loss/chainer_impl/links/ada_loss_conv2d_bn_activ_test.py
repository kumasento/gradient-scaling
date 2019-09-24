import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.chainer_impl.links.ada_loss_conv2d_bn_activ import AdaLossConv2DBNActiv
from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling


class AdaLossConv2DBNActivTest(unittest.TestCase):
    """ """

    def test_backward(self):
        with chainer.using_config('dtype', chainer.mixed16):
            x = chainer.Variable(
                np.random.normal(size=(1, 3, 4, 4)).astype('float16'))
            link = chainer.Sequential(
                AdaLossConv2DBNActiv(3,
                                     4,
                                     ksize=3,
                                     ada_loss_cfg={'fixed_loss_scale': 2}),
                lambda x: loss_scaling(x, 16.),
            )
            y = link(x)
            y.grad = np.ones_like(y.array, dtype=np.float16)
            y.backward()

        # grad_var can propagate
        self.assertTrue(hasattr(x.grad_var, 'loss_scale'))
        self.assertTrue(getattr(x.grad_var, 'loss_scale'), 2 * 16)


testing.run_module(__name__, __file__)
