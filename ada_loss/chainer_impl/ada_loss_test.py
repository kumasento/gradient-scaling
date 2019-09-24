import unittest
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import testing

from ada_loss.chainer_impl.ada_loss import AdaLossChainer

np.random.seed(0)


class AdaLossChainerTest(unittest.TestCase):
    """ Test the AdaLossChainerLinear. """

    def test_element_wise_multiply(self):
        """ Element-wise multiplication """
        ada_loss = AdaLossChainer()

        g = chainer.Variable(np.random.normal(size=(2, 2)).astype('float32'))
        W = chainer.Variable(np.random.normal(size=(2, 2)).astype('float32'))
        r = ada_loss.get_element_wise_multiply(g, W)

        # expected sequence
        self.assertEqual(r[0], g.data[0, 0] * W.data[0, 0])
        self.assertEqual(r[1], g.data[0, 1] * W.data[1, 0])
        self.assertEqual(r[2], g.data[0, 0] * W.data[0, 1])
        self.assertEqual(r[3], g.data[0, 1] * W.data[1, 1])
        self.assertEqual(r[4], g.data[1, 0] * W.data[0, 0])
        self.assertEqual(r[5], g.data[1, 1] * W.data[1, 0])
        self.assertEqual(r[6], g.data[1, 0] * W.data[0, 1])
        self.assertEqual(r[7], g.data[1, 1] * W.data[1, 1])

        # manually insert zeros into g and W
        g.data[0, 0] = 0
        W.data[1, 1] = 0
        r = ada_loss.get_element_wise_multiply(g, W)
        self.assertEqual(len(r), 4)
        r = ada_loss.get_element_wise_multiply(g, W, filter_zero=False)
        self.assertEqual(len(r), 8)

    def test_get_loss_scale_by_element_wise_range(self):
        """ Test how the loss scale works """
        u_max, u_min = 1e3, 1e-3
        ada_loss = AdaLossChainer(u_max=u_max, u_min=u_min)

        g = chainer.Variable(np.random.normal(size=(32, 32)).astype('float32'))
        W = chainer.Variable(np.random.normal(size=(32, 32)).astype('float32'))
        s = ada_loss.get_loss_scale_by_element_wise_range(g, W)

        # no overflow will happen
        self.assertTrue((np.dot(g.array, W.array) * s < u_max).all())

        # will overflow
        g = chainer.Variable(
            np.random.normal(scale=16, size=(32, 32)).astype('float32'))
        W = chainer.Variable(
            np.random.normal(scale=16, size=(32, 32)).astype('float32'))
        s = ada_loss.get_loss_scale_by_element_wise_range(g, W)

        self.assertLessEqual(s, 1.0)
        # no overflow will happen still
        self.assertTrue((np.dot(g.array, W.array) * s < u_max).all())

    def test_get_prev_scale(self):
        """ Check how prev_scale is implemented.
            Should extract correctly the loss_scale from the previous layer. """
        g = chainer.Variable(np.random.normal(size=1).astype('float32'))
        g.__dict__['loss_scale'] = 2.0

        ada_loss = AdaLossChainer()
        self.assertEqual(ada_loss.get_prev_scale(g), 2.0)

    def test_unscaled_gradient(self):
        """ Check whether the unscaled performs correctly. """
        g_data = np.random.normal(size=16).astype('float32')
        g = chainer.Variable(g_data)

        ada_loss = AdaLossChainer(dtype='float32', debug_level=1)
        ug = ada_loss.get_unscaled_gradient(g, 2.0)
        self.assertTrue(np.allclose(ug.array * 2.0, g_data))

        # float16
        g_data = np.random.normal(size=16).astype('float16') * 100
        g = chainer.Variable(g_data)
        ada_loss = AdaLossChainer(dtype='float16', debug_level=1)
        ug = ada_loss.get_unscaled_gradient(g, np.array(2.0, dtype='float32'))
        self.assertTrue(np.allclose(ug.array * 2.0, g_data))
        # cause overflow
        loss_scale = 1e-6
        with self.assertRaises(ValueError):
            ada_loss.get_unscaled_gradient(
                g, np.array(loss_scale, dtype='float32'))

    def test_get_scaled_gradient(self):
        """ Test the scaling """
        g = chainer.Variable(np.random.normal(size=1).astype('float32'))
        scale = 2.0

        # NOTE: float32 is necessary
        ada_loss = AdaLossChainer(dtype='float32')
        s_grad = ada_loss.get_scaled_gradient(g, scale)
        self.assertTrue(np.allclose(g.array * 2, s_grad.array))
        self.assertEqual(getattr(s_grad, 'loss_scale'), 2.0)

    def test_power_of_two_in_get_loss_scale(self):
        """ Check the switch of power_of_two """
        # turn ON
        dtype = np.float16
        ada_loss = AdaLossChainer(dtype=dtype,
                                  loss_scale_method='element_wise_range',
                                  use_bound=False)
        g = chainer.Variable(np.array([[1e-5]], dtype=dtype))
        W = chainer.Variable(np.array([[1e-4]], dtype=dtype))
        s = ada_loss.get_loss_scale(g, W)
        self.assertEqual(s, 32)

        # turn OFF
        ada_loss = AdaLossChainer(dtype=dtype,
                                  loss_scale_method='element_wise_range',
                                  power_of_two=False,
                                  use_bound=False)
        g = chainer.Variable(np.array([[1e-5]], dtype=dtype))
        W = chainer.Variable(np.array([[1e-4]], dtype=dtype))
        s = ada_loss.get_loss_scale(g, W)
        self.assertFalse(s == 32)

    def _test_scaled_grad(self, scale_val, dtype, prev_scale):
        g_data = np.random.normal(size=16).astype(dtype)
        g = chainer.Variable(g_data)
        scale = np.array(scale_val, dtype=dtype)

        ada_loss = AdaLossChainer(dtype=dtype)
        sg = ada_loss.get_scaled_gradient(g, scale, prev_scale=prev_scale)

        self.assertTrue(np.allclose(sg.array, g_data * scale_val))
        self.assertEqual(ada_loss.grad_loss_scale(sg), scale_val * prev_scale)

    def test_scaled_gradient(self):
        """ Test how the scaling works. """
        self._test_scaled_grad(2.0, np.float16, 1.0)
        self._test_scaled_grad(2.0, np.float16, 2.0)

        with self.assertRaises(AssertionError):
            self._test_scaled_grad(1e4, np.float16, 1e4)

    # approx method
    def test_get_mean_and_std(self):
        """ """
        dtype = np.float16
        x_data = np.random.normal(size=16).astype(dtype)
        x = chainer.Variable(x_data)
        ada_loss = AdaLossChainer(dtype=dtype)
        mu, sigma = ada_loss.get_mean_and_std(x)

        self.assertTrue(np.allclose(mu, x_data.astype(np.float32).mean()))
        self.assertTrue(np.allclose(sigma, x_data.astype(np.float32).std()))
        self.assertEqual(mu.dtype, np.float32)
        self.assertEqual(sigma.dtype, np.float32)

        # test numerical issue
        ada_loss = AdaLossChainer(dtype=dtype, debug_level=1)
        with self.assertRaises(AssertionError):
            x_data[0] = np.nan
            mu, sigma = ada_loss.get_mean_and_std(x)

    def test_get_mean_and_std_of_product(self):
        dtype = np.float16
        X_data = np.random.normal(size=16).astype(dtype)
        Y_data = np.random.normal(size=16).astype(dtype)
        X = chainer.Variable(X_data)
        Y = chainer.Variable(Y_data)

        ada_loss = AdaLossChainer(dtype=dtype)
        mu, sigma = ada_loss.get_mean_and_std_of_product(X, Y)

        X_mu, X_sigma = (X_data.astype(np.float32).mean(),
                         X_data.astype(np.float32).std())
        Y_mu, Y_sigma = (Y_data.astype(np.float32).mean(),
                         Y_data.astype(np.float32).std())

        self.assertEqual(mu.dtype, np.float32)
        self.assertEqual(sigma.dtype, np.float32)

        self.assertTrue(np.allclose(X_mu * Y_mu, mu))
        self.assertTrue(
            np.allclose(
                np.sqrt((X_sigma**2 + X_mu**2) * (Y_sigma**2 + Y_mu**2) -
                        (X_mu * Y_mu)**2).astype(np.float32), sigma))

    def _test_get_loss_scale_by_approx_range(self, g_sigma, W_sigma, dtype):
        g_data = np.random.normal(scale=g_sigma, size=(32, 32)).astype(dtype)
        W_data = np.random.normal(scale=W_sigma, size=(32, 32)).astype(dtype)
        g = chainer.Variable(g_data)
        W = chainer.Variable(W_data)

        ada_loss = AdaLossChainer(dtype=dtype, debug_level=1)

        scale = ada_loss.get_loss_scale_by_approx_range(g, W)
        self.assertEqual(scale.dtype, ada_loss.full_dtype)

        nnz1 = np.count_nonzero(np.dot(g_data, W_data))
        nnz2 = np.count_nonzero(np.dot(scale * g_data, W_data))
        if scale > 1:  # scaling is effective
            self.assertTrue(nnz1 < nnz2)
            self.assertFalse(np.isinf(np.dot(scale * g_data, W_data)).any())
        elif scale == 1:  # scaling has no effect
            self.assertTrue(nnz1 == nnz2)
        else:  # prevent overflow
            self.assertTrue(np.isinf(np.dot(g_data, W_data)).any())
            self.assertFalse(np.isinf(np.dot(scale * g_data, W_data)).any())

    def test_get_loss_scale_by_approx_range(self):
        """ """
        np.random.seed(0)
        self._test_get_loss_scale_by_approx_range(1e-6, 1e-6, np.float16)
        self._test_get_loss_scale_by_approx_range(1e-3, 1e-3, np.float16)
        self._test_get_loss_scale_by_approx_range(1e-2, 1e-2, np.float16)
        self._test_get_loss_scale_by_approx_range(100, 100, np.float16)
        self._test_get_loss_scale_by_approx_range(10000, 10000, np.float16)

    def test_bound_loss_scale_by_norm(self):
        """ """
        # g_in = np.random.normal(size=(32, 32)).astype(np.float32)
        # W = np.random.normal(size=(32, 32)).astype(np.float32)
        # g_out = np.dot(W, g_in)
        # _, s, _ = np.linalg.svd(W)
        # print(s, np.linalg.norm(W))
        # print(np.linalg.norm(W) * np.linalg.norm(g_in))
        # print(np.linalg.norm(g_in) / np.linalg.norm(np.linalg.inv(W)))
        # print(np.linalg.norm(W) / np.linalg.norm(np.linalg.inv(g_in)))
        # print(np.linalg.norm(g_out))

        # assert False

    def test_bound_loss_scale_by_heuristics(self):
        """ Heuristics testing """

    def test_rescaling(self):
        """ Test the rescaling method """
        # case 1: none scale will cause overflow
        gs = [
            chainer.Variable(np.random.normal(size=16)),
            chainer.Variable(np.random.normal(size=16)),
        ]
        gs[0].__dict__['loss_scale'] = 1
        gs[1].__dict__['loss_scale'] = 2

        ada_loss = AdaLossChainer()
        gs = ada_loss.rescaling(gs)

        # scale to the larger one
        self.assertEqual(gs[0].__dict__['loss_scale'], 2)

        # case 2: now we have overflow problem
        gs = [
            chainer.Variable(np.random.normal(size=16)),
            chainer.Variable(np.random.normal(size=16)),
        ]
        gs[0].__dict__['loss_scale'] = 65536
        gs[1].__dict__['loss_scale'] = 2
        gs = ada_loss.rescaling(gs)
        self.assertEqual(gs[0].__dict__['loss_scale'], 2)


testing.run_module(__name__, __file__)