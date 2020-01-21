from ..ada_loss import AdaLoss

import functools
import math
from timeit import default_timer as timer

import numpy as np
from scipy.special import erfinv
import chainer
import chainer.functions as F
from chainer import utils


class AdaLossChainer(AdaLoss):
    """ The implementation of the adaptive loss scaling method in Chainer """

    _loss_scale_key = 'loss_scale'

    def get_fan_in(self, W):
        if len(W.shape) == 2:
            return np.array(W.shape[0], dtype=self.full_dtype).item()
        else:
            return np.array(W.shape[0] * W.shape[2] * W.shape[3],
                            dtype=self.full_dtype).item()  # deconv

    def get_matrices_to_be_multiplied(self, g, W):
        """ Need to preprocess inputs to turn them into matrices  """
        g_, W_ = g.array, W.array
        if len(g_.shape) == 2 and len(W_.shape) == 2:
            return g_, W_
        if len(g_.shape) == 4 and len(W_.shape) == 4:
            # NOTE: assume this is convolution
            F = g_.shape[1]
            new_g_ = g_.transpose([1, 0, 2, 3]).reshape([F, -1]).T
            new_W_ = W_.reshape([F, -1])
            return new_g_, new_W_

        raise ValueError(
            'g and W shape cannot be transformed to matrices: {} and {}'.format(
                g.shape, W.shape))

    def get_element_wise_multiply(self,
                                  g,
                                  W,
                                  filter_zero=True,
                                  cast_to_fp32=True):
        """ Get the result for element-wise multiplying g and W """
        g_, W_ = self.get_matrices_to_be_multiplied(g, W)

        xp = chainer.backend.get_array_module(g_)
        if xp.isnan(g_).any():
            raise ValueError('Input gradient contains NaN')
        if xp.isnan(W_).any():
            raise ValueError('Input weight contains NaN')

        N, K = g_.shape
        M = W_.shape[1]

        # element-wise
        ge_ = g_.reshape((N, 1, K)).repeat(M, axis=1).flatten()
        We_ = W_.T.reshape((1, M, K)).repeat(N, axis=0).flatten()
        # NOTE: this may not be necessary for fp32 update
        if cast_to_fp32:
            ge_, We_ = ge_.astype('float32'), We_.astype('float32')

        # run multiplication
        gW_ = xp.multiply(ge_, We_)

        if filter_zero:  # if one operand is 0, filter out result
            nonzero_mask = xp.bitwise_and(ge_ != 0, We_ != 0)
            gW_ = gW_[nonzero_mask]

        return gW_

    def get_loss_scale_by_element_wise_range(self,
                                             g,
                                             W,
                                             max_rescue_ratio=None,
                                             cast_to_fp32=True,
                                             bound_by_fan_in=True,
                                             norm_upper_bound=1024,
                                             scale_upper_bound=16,
                                             prev_scale=None):
        """ Scale the loss by the given range. """
        assert cast_to_fp32, 'We only support FP32 mode element-wise analysis'
        assert self.range[0] is not None and self.range[1] is not None

        # get element-wise results for further analysis
        res = self.get_element_wise_multiply(g, W, cast_to_fp32=cast_to_fp32)
        # prevent further issues
        if len(res) == 0:
            # NOTE: temporarily do nothing
            # return 1.0,
            print('g = \n{}'.format(g.array))
            print('W = \n{}'.format(W.array))
            raise RuntimeError(
                'There should be non-zero result from multiplying g and W element-wise, got: {}'
                .format(len(res)))

        xp = chainer.backend.get_array_module(res)

        # take absolute
        res = xp.abs(res)
        if xp.isnan(res).any():
            raise ValueError(
                'NaN appeared in the element-wise multiplication result.')

        # NOTE: we don't need to scale down to self.range[0], just
        # to promote values out of it
        # print(self.range[0], (self.range[0] < res).all())
        min_scales = xp.maximum(self.range[0] / res, 1.0)

        # max rescue ratio will limit the amount of elements to be rescued
        # NOTE: may not be very effective
        if isinstance(max_rescue_ratio, float):
            min_scales = xp.sort(min_scales)
            max_n_rescue = int(len(min_scales) * max_rescue_ratio)
            min_scales = min_scales[:max_n_rescue]

        # It could have values below 1. In this case, we already
        # have overflow and it should be prevented with higher priority.
        max_scales = self.range[1] / res

        # NOTE: to prevent overflow while accumulating multiplication results
        # we may need to bound the scaling factor by the fan_in.
        # It is a quite strict bound since there might exist negative values
        # and cancels accumulated values.
        if bound_by_fan_in:
            max_scales /= self.get_fan_in(W)
        # print(min_scales.max(), max_scales.min())

        # NOTE: we don't need to scale more than the one that can
        # move all underflow values out of range. And causing overflow
        # would be avoided.
        loss_scale = xp.minimum(min_scales.max(), max_scales.min())
        self.check_scale(loss_scale)

        if hasattr(xp, 'asnumpy'):
            loss_scale = xp.asnumpy(loss_scale)

        return loss_scale

    def get_mean_and_std(self, X, lognormal=False):
        """ Get the mean and variance of a chainer variable """
        X_ = X.array

        xp = chainer.backend.get_array_module(X_)
        if self.debug_level >= 1:
            assert not xp.isnan(X_).any() and not xp.isinf(X_).any()

        # X_ = X_[xp.logical_and(X_ < threshold, X_ > - threshold)]

        if X.dtype != self.full_dtype:
            # Now X_ and X.array are two different things
            X_ = X_.astype(self.full_dtype)
        if lognormal:
            X_ = xp.log(xp.abs(X_[X_ != 0]))

        mu, sigma = X_.mean(), X_.std()

        if hasattr(xp, 'asnumpy'):
            mu = xp.asnumpy(mu)
            sigma = xp.asnumpy(sigma)

        if X.dtype != self.full_dtype:
            del X_

        return mu, sigma

    def get_mean_and_std_of_product(self, X, Y):
        """ Use the rule for product distribution """
        X_mu, X_sigma = self.get_mean_and_std(X)
        Y_mu, Y_sigma = self.get_mean_and_std(Y)

        # O_mu = X_mu * Y_mu
        O_var = ((X_sigma**2 + X_mu**2) * (Y_sigma**2 + Y_mu**2)) # - (O_mu)**2)

        if self.debug_level >= 1 and (O_var < 0 or np.isnan(O_var) or
                                      np.isinf(O_var)):
            raise ValueError(
                'Cannot compute the product stddev, got o_var {} X_mu {} X_sigma {} Y_mu {} Y_sigma {}'
                .format(O_var, X_mu, X_sigma, Y_mu, Y_sigma))

        O_sigma = np.sqrt(O_var).astype(self.full_dtype)
        if self.debug_level >= 1:
            assert not np.isnan(O_sigma) and not np.isinf(O_sigma)

        return 0, O_sigma

    def get_loss_scale_by_approx_range(self,
                                       g,
                                       W=None,
                                       n_sigma=1e-2,
                                       lognormal=False,
                                       bound_by_fan_in=True):
        """ """
        xp = chainer.backend.get_array_module(g)
        # NOTE assume zero mean
        u_min, u_max = self.range

        # calculate the statistics
        calc_stat_start = timer()
        if W is not None:
            # lognormal won't be applied in this branch
            _, o_sigma = self.get_mean_and_std_of_product(g, W)
        else:
            o_mu, o_sigma = self.get_mean_and_std(g, lognormal=lognormal)
        calc_stat_end = timer()
        if self.profiler is not None:
            self.profiler.add_time('calc_stat', calc_stat_end - calc_stat_start)

        # NOTE: it is possible that o_sigma is 0.
        # e.g., a all-zero gradient.
        if o_sigma == 0:
            return np.array(1, dtype=self.full_dtype)

        # NOTE to prevent overflow
        g_min, g_max = (g.array.min().astype(self.full_dtype),
                        g.array.max().astype(self.full_dtype))
        if hasattr(xp, 'asnumpy'):
            g_min = xp.asnumpy(g_min)
            g_max = xp.asnumpy(g_max)
        o_max = np.abs([g_max, g_min]).max()

        if W is not None:
            # print('o_sigma = {} {}'.format(o_sigma, o_sigma.dtype))
            # print('o_mu={}'.format(g_mu * w_mu))
            W_min, W_max = (W.array.min().astype(self.full_dtype),
                            W.array.max().astype(self.full_dtype))
            if hasattr(xp, 'asnumpy'):
                W_min = xp.asnumpy(W_min)
                W_max = xp.asnumpy(W_max)

            # if self.debug_level >= 1:
            #     assert not xp.isnan(g_max) and not xp.isinf(g_max)
            #     assert not xp.isnan(W_max) and not xp.isinf(W_max)

            if bound_by_fan_in:
                u_max /= self.get_fan_in(W)

            o_max = np.abs(
                np.array(
                    [g_max * W_max, g_min * W_min, g_max * W_min,
                     g_min * W_max])).max()
            # NOTE: assuming the existence of ReLU 
            o_sigma *= np.sqrt(0.5 * self.get_fan_in(W))

        # NOTE: need to cast n_sigma
        n_sigma = np.array(n_sigma, dtype=self.full_dtype)

        # TODO: refactorize 
        if lognormal and W is None:
            loss_scale = np.exp(
                    np.log(u_min.astype(self.full_dtype)) -
                    o_mu -
                    o_sigma * np.sqrt(2) * erfinv(2 * self.n_uf - 1).astype(self.full_dtype))
            # assert not np.isnan(loss_scale) and not np.isinf(loss_scale)
        else:
            loss_scale = u_min / (self.n_uf_threshold * o_sigma)
        # constrain to 1.
        loss_scale = np.maximum(loss_scale, np.array(1.0,
                                                     dtype=self.full_dtype))
        # print('min loss_scale={}'.format(loss_scale))

        if self.debug_level >= 1:
            assert not np.isnan(o_max) and not np.isinf(o_max)

        upper_scale = u_max / o_max
        loss_scale = np.minimum(loss_scale, upper_scale)
        loss_scale = loss_scale.astype(self.full_dtype)

        # if hasattr(xp, 'asnumpy'):
        #     loss_scale = xp.asnumpy(loss_scale)

        return loss_scale

    def get_loss_scale_by_abs_range(self,
                                    g,
                                    W,
                                    bound_by_fan_in=True,
                                    o_min_epsilon=1e-15):
        """ """
        xp = chainer.backend.get_array_module(g)
        g_, W_ = xp.abs(g.array), xp.abs(W.array)
        g_, W_ = g_[g_ != 0], W_[W_ != 0]

        # NOTE: it is possible that the number of nonzero in W_ is 0 due to zero-initialized weight
        # in that case, we will skip this calculation
        if g_.size == 0 or W_.size == 0:
            return np.array(1.0, dtype=self.full_dtype)

        g_min, g_max = (g_.min().astype(self.full_dtype),
                        g_.max().astype(self.full_dtype))
        W_min, W_max = (W_.min().astype(self.full_dtype),
                        W_.max().astype(self.full_dtype))

        mm = np.array([g_min, W_min, g_max, W_max], dtype=self.full_dtype)
        bd = np.array(
            [mm[0] * mm[1], mm[0] * mm[3], mm[1] * mm[2], mm[3] * mm[2]],
            dtype=self.full_dtype)
        if np.isnan(bd).any() or np.isinf(bd.any()):
            raise ValueError(
                'candidates contains invalid values: bd={} mm={}'.format(
                    bd, mm))
        o_min, o_max = bd.min(), bd.max()
        o_min = np.maximum(o_min, o_min_epsilon)

        u_min, u_max = self.range
        if bound_by_fan_in:
            u_min /= self.get_fan_in(W)
            u_max /= self.get_fan_in(W)

        # print(o_min, o_max, u_min / o_min, u_max / o_max)
        lower = np.maximum(u_min / o_min, 1.0)
        scale = np.minimum(lower, u_max / o_max)
        self.check_scale(scale)

        scale = np.array(scale, dtype=self.full_dtype)
        # print(u_min / o_min, u_max / o_max)
        return scale

    # def get_loss_scale_by_approx_range_v0(self,
    #                                       g,
    #                                       W,
    #                                       n_sigma=2,
    #                                       bound_by_fan_in=True,
    #                                       o_min_epsilon=1e-15,
    #                                       o_max_by_maximum=False):
    #     """ Implement this method in Chainer. """
    #     xp = chainer.backend.get_array_module(g)
    #     g_mu, g_sigma = self.get_mean_and_variance(g)
    #     w_mu, w_sigma = self.get_mean_and_variance(W)

    #     # o_mu = g_mu * w_mu
    #     # formula for multiplying two independent random variables
    #     o_sigma = xp.sqrt((g_sigma**2 + g_mu**2) * (w_sigma**2 + w_mu**2) -
    #                       (g_mu * w_mu)**2)

    #     # the formula for half-normal distribution (we assume that)
    #     o_mu = o_sigma * xp.sqrt(2 / xp.pi)
    #     o_sigma = xp.sqrt((o_sigma**2) * (1 - 2 / xp.pi))

    #     # cast the data type
    #     o_mu = xp.array(o_mu, dtype=self.full_dtype)
    #     o_sigma = xp.array(o_sigma, dtype=self.full_dtype)

    #     # how the approximation works
    #     # we first calculate the range by the estimated mean and sigma
    #     o_min = o_mu - n_sigma * o_sigma

    #     # it's possible that o_min is smaller than 0
    #     # we cast it back to a value above 0
    #     o_min = xp.maximum(o_min, o_min_epsilon)

    #     # the maximum value should be the exact maximum
    #     if o_max_by_maximum:
    #         o_max = (xp.abs(g.array).max().astype(self.full_dtype) *
    #                  xp.abs(W.array).max().astype(self.full_dtype))
    #     else:
    #         o_max = o_mu + n_sigma * o_sigma

    #     # get the expected range
    #     u_min, u_max = self.range
    #     if bound_by_fan_in:
    #         u_max /= self.get_fan_in(w)

    #     # constrain the underflow scaler
    #     uf_scale = xp.maximum(u_min / o_min, 1.0)
    #     of_scale = u_max / o_max
    #     # print('o_min={:10.6f} o_max={:10.6f} uf={:10.6f} of={:10.6f}'.format(
    #     #     o_min.item(), o_max.item(), uf_scale.item(), of_scale.item()))

    #     # NOTE: min(scale all underflow, scale one overflow)
    #     scale = xp.minimum(uf_scale, of_scale)

    #     # print(o_min, o_max, max(self.range[0] / o_min, self.range[1] / o_min))

    #     if hasattr(xp, 'asnumpy'):
    #         scale = xp.asnumpy(scale)
    #     return scale

    def check_scale(self, scale):
        """ Sanity checking scale value """
        xp = chainer.backend.get_array_module(scale)
        if xp.isnan(scale) or xp.isinf(scale):
            raise ValueError('scale value {} is invalid'.format(scale))

    def check_grad_var(self, grad_var):
        """ Sanity checking the gradient """
        if not hasattr(grad_var, self._loss_scale_key):
            raise ValueError('grad_var should contain the key for loss scale')

    def check_grad(self, g):
        """ Check the gradient array.
            Should not contain `inf` or `nan` values.
        """
        xp = chainer.backend.get_array_module(g)
        if xp.isnan(g).any():
            raise ValueError('Gradient of shape {} has NaN element'.format(
                g.shape))
        if xp.isinf(g).any():
            raise ValueError('Gradient of shape {} has Inf element'.format(
                g.shape))

    def bound_loss_scale_by_norm(self, scale, W):
        """ We want that the loss scale value won't change a lot the
            norm of the activation gradient.

            ||g_out|| = scale x ||g_in x W|| ~= scale x ||W|| x ||g_in||
            so that we want scale x ||W|| to be at most 1.
        """

    def bound_loss_scale_by_heuristics(self, loss_scale, W, prev_scale=None):
        """ Implemented in Chainer """
        self.check_scale(loss_scale)
        # bound the scale by prev_scale and norm
        if prev_scale is None:
            prev_scale = np.array(1.0, dtype=self.scale_dtype)

        # # bound by norm (NOT VERY HELPFUL and IT IS WRONG)
        # # calculate norm
        # xp = chainer.backend.get_array_module(W)
        # loss_scale = xp.array(loss_scale)
        # W_norm = xp.linalg.norm(W.array.astype(self.full_dtype))
        # # bound loss scale
        # # NOTE: I need to pass norm_upper_bound to cupy or there will be type mismatching
        # loss_scale = xp.minimum(
        #     xp.array(self.norm_upper_bound) / (xp.array(prev_scale) * W_norm),
        #     loss_scale)
        # self.check_scale(loss_scale)

        # if self.debug_level >= 1:
        #     if xp.isnan(W_norm * prev_scale) or xp.isinf(W_norm * prev_scale):
        #         print(W.array)
        #         raise ValueError(
        #             'W_norm {} times prev scale {} result in {}'.format(
        #                 W_norm, prev_scale, W_norm * prev_scale))
        # loss_scale = xp.asnumpy(loss_scale)

        # BOUND BY ACCUMULATED LOSS SCALE
        accum_bound = (np.array(self.accum_upper_bound, dtype=self.full_dtype) /
                       np.array(prev_scale.item()))
        loss_scale = np.minimum(accum_bound, loss_scale)

        # HARD upper bound
        loss_scale = np.minimum(
            np.array(self.scale_upper_bound, dtype=self.full_dtype),
            loss_scale).astype(self.full_dtype)
        if self.debug_level >= 1:
            self.check_scale(loss_scale)

        return loss_scale

    def get_prev_scale(self, g):
        """ Get the previous scale.
            scale_map is placed on the CPU. 
        """
        # if self.node_id == len(self.scale_map) - 2:
        #     self.scale_map[self.node_id + 1] = self.scale_map[self.node_id + 1]

        # prev_scale = np.product(self.scale_map[(self.node_id + 1):])
        # self.check_scale(prev_scale)

        # # Sanity check
        # # NOTE: we need to ensure that prev_scale can be correctly calculated and
        # # passed to the weight update
        # if np.isnan(prev_scale.astype(self.dtype)) or np.isinf(
        #         prev_scale.astype(self.dtype)):
        #     raise ValueError(
        #         'prev_scale cannot be correctly casted to {}: {} ({}) {} ({})'.
        #         format(self.dtype, prev_scale, self.scale_dtype,
        #                prev_scale.astype(self.dtype), self.dtype))

        # NOTE: the loss scale should be propagated alongside with g
        self.check_grad_var(g)
        return self.grad_loss_scale(g)

    def get_unscaled_gradient(self, g, prev_scale, dtype=None):
        """ Scale back gradient by the previous scale
        Args:
            g: gradient array
            prev_scale: the scale to be divided
            dtype: cast the result gradient
        Returns:
            A gradient variable
        """
        xp = chainer.backend.get_array_module(g)
        prev_scale = xp.array(prev_scale)

        # NOTE: here we optionally change the dtype of the gradient
        if dtype is None:
            dtype = self.dtype
        g_ = g.array
        if not self.power_of_two:
            g_ = g_.astype(self.full_dtype)
            prev_scale = prev_scale.astype(self.full_dtype)

        ug_ = g_ / prev_scale
        if self.debug_level >= 1:
            # sanity check the unscaled gradient
            self.check_grad(ug_.astype(dtype))

        ug = chainer.Variable(ug_.astype(dtype))
        # unscaled, maybe not necessary
        self.set_loss_scale(ug, xp.array(1., dtype=self.scale_dtype).item())

        return ug

    def get_scaled_gradient(self, g, scale, prev_scale=1., dtype=None):
        """ Scale gradient """
        xp = chainer.backend.get_array_module(g)

        # NOTE: here we optionally change the dtype of the gradient
        if dtype is None:
            dtype = g.dtype
        # change the type of scale
        scale = xp.array(scale, dtype=self.scale_dtype)
        prev_scale = xp.array(prev_scale, dtype=self.scale_dtype)

        g_ = g.array
        if not self.power_of_two:
            g_ = g_.astype(self.full_dtype)
            scale = scale.astype(self.full_dtype)
            prev_scale = prev_scale.astype(self.full_dtype)

        # assert scale.dtype == g_.dtype, 'Scale dtype={} does not match grad dtype={}'.format(scale.dtype, g_.dtype)
        if scale.dtype != g_.dtype:
            if g_.dtype == np.float32:
                scale = scale.astype(g_.dtype)
            else:
                raise RuntimeError('Scale dtype={} does not match grad dtype={}'.format(scale.dtype, g_.dtype))

        sg_ = g_ * scale
        if self.debug_level >= 1:
            # sanity check the unscaled gradient
            self.check_grad(sg_.astype(dtype))

        sg = chainer.Variable(sg_.astype(dtype))

        # NOTE: updated, we need to attach the new scale
        scale_ = scale * prev_scale
        # print(scale, prev_scale, scale_)
        assert not xp.isnan(scale_) and not xp.isinf(scale_)

        self.set_loss_scale(sg, scale_)

        return sg

    def set_loss_scale(self, g, scale):
        g.__dict__[self._loss_scale_key] = scale

    def grad_loss_scale(self, g):
        return g.__dict__[self._loss_scale_key]

    def record_loss_scale(self, key, val):
        """ Record loss scale by an AdaLossRecorder """
        if self.recorder is None:
            return  # Skipped
        self.recorder.record(key, val, label=self.func.label)

    def rescaling(self, gs):
        """ The implementation of rescaling in chainer.
            This API takes a list of gradients and rescales all of them
            to the same scale value.

            Args:
                gs(list): a list of gradients
            Returns:
                A list of rescaled gradients
        """
        assert len(gs) >= 2, 'Number of gradients should be >= 2'
        scales = [self.grad_loss_scale(g) for g in gs]
        scales = list(sorted(scales))
        target = None

        if not self.is_updating:
            target = self.loss_scales[-1]

        else:
            # search from the largest to the smallest
            max_gs = []
            for g in gs:
                xp = chainer.backend.get_array_module(g.array)
                max_gs.append(
                        xp.max(xp.abs(g.array)).astype(self.full_dtype) /
                        self.grad_loss_scale(g).astype(self.full_dtype))

            for scale in scales[::-1]:
                is_overflow = False
                # check whether this scale will not cause overflow
                if max(max_gs) * scale < self.range[1]:
                    target = scale
                    break

            # TODO: take care of this case
            # TODO: refactorize
            if target is None:
                # temporarily, we search for smaller scales
                scale_ = min(scales)
                while scale_ >= 1 / 8:
                    if max(max_gs) * scale_ < self.range[1]:
                        break
                    scale_ /= 2
                target = scale_
            if max(max_gs) * target >= self.range[1]:
                raise ValueError('Cannot find a suitable target for max gradients: {}'.format(max_gs))
            self.loss_scales.append(target)

        # print(target)

        # run rescaling
        for g in gs:
            scale = self.grad_loss_scale(g)
            scale_factor = target / scale
            g.array *= scale_factor
            self.set_loss_scale(g, target)

        return gs

