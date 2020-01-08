""" Define the class for adaptive loss scaling. """

import numpy as np
from scipy.special import erfinv
from timeit import default_timer as timer


class AdaLoss(object):
    """ Implementation of the adaptive loss scaling method. """

    def __init__(self,
                 func_params=None,
                 sanity_checker=None,
                 u_min=None,
                 u_max=None,
                 power_of_two=True,
                 loss_scale_method=None,
                 fixed_loss_scale=1.,
                 scale_upper_bound=16,
                 accum_upper_bound=1024,
                 max_rescue_ratio=None,
                 n_sigma=1e-3,
                 n_uf_threshold=1e-3,
                 debug_level=0,
                 use_bound=True,
                 dtype='float16',
                 full_dtype='float32',
                 func=None,
                 update_per_n_iteration=1,
                 recorder=None,
                 profiler=None):
        """ CTOR. """
        # parameters for the function, maybe useful
        if func_params is None:
            func_params = {}
        self.func_params = func_params

        # sanity checker for the performance.
        self.sanity_checker = sanity_checker

        self.dtype = dtype
        self.full_dtype = full_dtype
        # NOTE: scale dtype is the same as dtype for accurate
        # reduction of multiplication.
        self.scale_dtype = dtype

        # setup loss scale method
        if loss_scale_method is None:
            loss_scale_method = 'approx_range'
        self.loss_scale_method = loss_scale_method

        # approx method
        self.n_sigma = n_sigma
        self.n_uf = n_uf_threshold
        self.n_uf_threshold = erfinv(n_uf_threshold) * np.sqrt(2)

        # NOTE: if power_of_two is True, we will try to locate
        # the scale factor to a value that's exactly a power of 2
        self.power_of_two = power_of_two

        # fixed loss scale
        self.fixed_loss_scale = fixed_loss_scale
        # range to be constrained
        self.range = self.get_range(dtype, u_min=u_min, u_max=u_max)
        # number of data to be rescued
        self.max_rescue_ratio = max_rescue_ratio
        # bound for scales
        self.use_bound = use_bound
        self.scale_upper_bound = np.array(scale_upper_bound,
                                          dtype=self.scale_dtype)
        self.accum_upper_bound = np.array(accum_upper_bound,
                                          dtype=self.scale_dtype)

        # debugging
        self.debug_level = debug_level

        # function node
        self.func = func

        # recording
        self.recorder = recorder

        # profile
        self.profiler = profiler

        # Configure update frequency
        # will be used when the current iteration skips loss scale calculation
        self.loss_scales = []  # stored loss scales
        self.cur_iter = 0  # current iteration
        self.update_per_n_iteration = update_per_n_iteration

    def get_range(self, dtype, u_min=None, u_max=None):
        if u_min is None:
            u_min = self.get_u_min(dtype)
        if u_max is None:
            u_max = self.get_u_max(dtype)
        return np.array([u_min, u_max], dtype=self.full_dtype)

    def get_u_min(self, dtype):
        if dtype == 'float16' or dtype == np.float16:
            return 6e-8
        elif dtype == 'float32' or dtype == np.float32:
            return 1e-23
        raise ValueError('dtype cannot be recognized: {}'.format(dtype))

    def get_u_max(self, dtype):
        if dtype == 'float16' or dtype == np.float16:
            return 6e4
        elif dtype == 'float32' or dtype == np.float32:
            return 1e23
        raise ValueError('dtype cannot be recognized: {}'.format(dtype))

    def get_prev_scale(self, g):
        raise NotImplementedError(
            'get_prev_scale should be implemented in a concrete class')

    def get_loss_scale_by_element_wise_range(self,
                                             g,
                                             W,
                                             max_rescue_ratio=None,
                                             cast_to_fp32=True,
                                             bound_by_fan_in=True,
                                             accum_upper_bound=1024,
                                             scale_upper_bound=16,
                                             prev_scale=None):
        raise NotImplementedError(
            'get_loss_scale_by_element_wise_range should be implemented')

    def get_loss_scale_by_approx_range(self, g, W, n_sigma=2):
        """ Calculate loss scale by an statistically approximated range.

            We assume that g and W are drawn from independent distributions.
        """
        raise NotImplementedError(
            'get_loss_scale_by_approx_range should be implemented')

    def get_loss_scale_by_abs_range(self, g, W):
        """ Using absolute range to bound loss scale """
        raise NotImplementedError(
            'get_loss_scale_by_abs_range should be implemented')

    def bound_loss_scale_by_heuristics(self,
                                       loss_scale,
                                       W=None,
                                       prev_scale=None):
        """ Heuristic methods to bound the estimated loss scales """
        raise NotImplementedError(
            'bound_loss_scale_by_heuristics should be implemented')

    #################################
    # Control update frequency
    #################################

    @property
    def is_updating(self):
        """ Check whether the current iteration needs to recalculate
            loss scale. """
        return self.cur_iter % self.update_per_n_iteration == 0

    def get_unbound_loss_scale(self, g, W=None, prev_scale=None):
        """ Get the loss scale for the current layer. """
        # TODO: Refactorize this piece
        if not self.is_updating:
            # return the previously stored loss scale
            loss_scale = self.loss_scales[-1]
        else:
            # print('==> Updating loss scale at iteration {}'.format(
            #     self.cur_iter))
            if self.loss_scale_method == 'element_wise_range':
                loss_scale = self.get_loss_scale_by_element_wise_range(
                    g,
                    W,
                    prev_scale=prev_scale,
                    max_rescue_ratio=self.max_rescue_ratio)
            elif self.loss_scale_method == 'approx_range':
                loss_scale = self.get_loss_scale_by_approx_range(
                    g, W, n_sigma=self.n_sigma)
            elif self.loss_scale_method == 'abs_range':
                loss_scale = self.get_loss_scale_by_abs_range(g, W)
            elif self.loss_scale_method == 'fixed':
                loss_scale = np.array(self.fixed_loss_scale,
                                      dtype=self.scale_dtype)
            else:
                raise ValueError(
                    'Cannot recognize loss scale method "{}"'.format(
                        self.loss_scale_method))
            self.loss_scales.append(loss_scale)

            # print(loss_scale)

        # update state
        # TODO: test this functionality
        self.cur_iter += 1

        return loss_scale

    def get_loss_scale(self, g, W=None, prev_scale=None):
        # preliminary results
        loss_scale = self.get_unbound_loss_scale(g, W=None, prev_scale=prev_scale)
        self.record_loss_scale('unbound', loss_scale)

        # if self.loss_scale_method == 'fixed':
        #     return loss_scale
        if self.use_bound:
            loss_scale = self.bound_loss_scale_by_heuristics(
                loss_scale, W=W, prev_scale=prev_scale)
            self.record_loss_scale('bound', loss_scale)
        if self.power_of_two:
            loss_scale = self.get_power_of_two_scale(loss_scale)
            self.record_loss_scale('power_of_two', loss_scale)

        loss_scale = loss_scale.astype(self.scale_dtype)
        self.record_loss_scale('final', loss_scale)
        # print(loss_scale)
        return loss_scale.item()

    def record_loss_scale(self, key, val):
        """ Will be called after the loss scale is collected. """
        raise NotImplementedError(
            'record_loss_scale should be implemented properly')

    def get_unscaled_gradient(self, g, prev_scale):
        """ Scale back gradient by the previous scale """
        raise NotImplementedError(
            'get_unscaled_gradient should be implemented in a concrete class')

    def get_scaled_gradient(self, g, scale, prev_scale=1.):
        """ Scale gradient """
        raise NotImplementedError(
            'get_scaled_gradient should be implemented in a concrete class')

    def get_power_of_two_scale(self, scale):
        """ Find a suitable power-of-two value of scale"""
        # NOTE: this function should be performed in full precision (FP32)
        assert np.ndim(scale) == 0

        log_scale = np.log2(scale)  # NOTE: keep the interface
        pot_scale = np.power(2, np.floor(log_scale))

        return pot_scale.astype(self.scale_dtype)  # cast back

    def loss_scaling(self, g, W=None):
        """ Called to calculate the loss scale.

            This API works specifically for GEMM based operations.

            Returns a scaled and unscaled gradient
            https://github.com/chainer/chainer/blob/24846ca945f9847bdc2b7c773fba30e9983c43a2/chainer/optimizer.py#L218-L219
        """
        total_start = timer()

        prev_scale = self.get_prev_scale(g)
        scale = self.get_loss_scale(g, W=W, prev_scale=prev_scale)
        # u_grad = self.get_unscaled_gradient(g, prev_scale)
        grad = self.get_scaled_gradient(g, scale, prev_scale=prev_scale)

        total_end = timer()
        if self.profiler is not None:
            self.profiler.add_time('total', total_end - total_start)

        return grad, prev_scale  # u_grad

    def loss_scaling_cast(self, x):
        """ Calculate the loss scale in the cast scenario """
        raise NotImplementedError('loss_scaling_cast has not been implemented') 

    def rescaling(self, gs):
        """ Rescale a list of gradients. """
        raise NotImplementedError('rescaling has not been implemented yet.')
