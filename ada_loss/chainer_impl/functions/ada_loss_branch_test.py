""" Check the correctness of AdaLossBranch """
import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.chainer_impl.ada_loss import AdaLossChainer
from ada_loss.chainer_impl.functions.ada_loss_branch import AdaLossBranch

np.random.seed(0)


class AdaLossBranchTest(unittest.TestCase):
    """ Check the result from ada_loss_branch """

    def test_forward(self):
        """ forward should act as normal """
        x = chainer.Variable(np.random.normal(size=16))
        n = 3
        ys = AdaLossBranch(n_branch=3).apply((x,))
        for i in range(n):
            self.assertTrue(np.allclose(x.array, ys[i].array))

    def test_backward(self):
        """ Check whether rescaling works properly """
        x = chainer.Variable(np.random.normal(size=16).astype(np.float16))
        n = 3
        ys = AdaLossBranch(n_branch=3).apply((x,))

        for i in range(n):
            ys[i].grad_var = chainer.Variable(
                np.random.normal(size=16).astype(np.float16)
            )
            ys[i].grad_var.__dict__["loss_scale"] = 2

        # NOTE: seems backward all gradients
        ys[0].backward()
        self.assertEqual(x.grad_var.__dict__["loss_scale"], 2)

        # Overflow case
        x = chainer.Variable(np.random.normal(size=16).astype(np.float16))
        ys = AdaLossBranch(n_branch=3).apply((x,))
        gs = [np.random.normal(size=16).astype(np.float16) for _ in range(n)]
        for i in range(n):
            ys[i].grad_var = chainer.Variable(gs[i])
            ys[i].grad_var.__dict__["loss_scale"] = 2

        ys[0].grad_var.__dict__["loss_scale"] = 65536
        ys[0].backward()
        self.assertEqual(x.grad_var.__dict__["loss_scale"], 2)

        golden = (gs[0] * np.float16(2 / 65536) + sum(gs[1:])).astype(np.float16)
        # lossen the error threshold
        self.assertTrue(np.allclose(x.grad, golden, atol=1e-3))


testing.run_module(__name__, __file__)
