""" Implements low-precision optimizers. """
import numpy as np

import chainer
from chainer import optimizers
from chainer import backend
from chainer.backends import cuda

# TODO: we assume the existence of CuPy
from chainer.backends.cuda import cupy as cp
from chainer.optimizers.momentum_sgd import MomentumSGDRule, _default_hyperparam


class _DeterministicRoundUpRule(MomentumSGDRule):
    """ Override the update rule to deterministically use
        the smallest magnitude value in low-precision.
        TODO: override CPU rules
        TODO: use stochastic methods
    """

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        kernel = cuda.elementwise(
            "T grad, T lr, T momentum",
            "T param, T v",
            """ T u = lr * grad;
                if (grad != 0 && lr != 0 && u == 0)
                    u = grad > 0 ? 6e-8 : -6e-8; 
                v = momentum * v - u;
                param += v;""",
            "momentum_sgd",
        )
        kernel(
            grad,
            self.hyperparam.lr,
            self.hyperparam.momentum,
            param.data,
            self.state["v"],
        )


class _StochasticRoundingRule(MomentumSGDRule):
    """ Override the original momentum update by a stochastic rounding
        method. """

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        kernel = cuda.elementwise(
            "T grad, T lr, T momentum, T prob",
            "T param, T v",
            """ T u = lr * grad;
                T eps = (T) 6e-8;
                T high = eps / lr;
                if (abs(u) >= prob && abs(u) < high)
                    u += u > 0 ? 6e-8 : -6e-8; 
                v = momentum * v - u;
                param += v;""",
            "momentum_sgd",
        )

        # create the variable for stochastic rounding
        epsilon = cp.float16(6e-08)
        low, high = epsilon, cp.float16(epsilon / self.hyperparam.lr)
        prob = cp.random.uniform(low=low, high=high, size=grad.shape).astype(cp.float16)

        kernel(
            grad,
            cp.float16(self.hyperparam.lr),
            cp.float16(self.hyperparam.momentum),
            prob,
            param.data,
            self.state["v"],
        )


class _ObserveZeroRule(MomentumSGDRule):
    """ Observe the number of zero updates """

    _kernel = None
    _nzu_kernel = None  # number of zero update

    def init_state(self, param):
        super(_ObserveZeroRule, self).init_state(param)

        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state["u"] = xp.zeros_like(param.data)
            self.state["nzu"] = 0

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        if _ObserveZeroRule._kernel is None:
            _ObserveZeroRule._kernel = cuda.elementwise(
                "T grad, T lr, T momentum",
                "T param, T v, T u",
                """u = lr * grad;
                   v = momentum * v - u;
                   param += v;""",
                "momentum_sgd",
            )
        if _ObserveZeroRule._nzu_kernel is None:
            _ObserveZeroRule._nzu_kernel = cuda.reduce(
                "T grad, T u",
                "int32 n",
                "grad != 0 & u == 0",
                "a + b",
                "n = a",
                "0",
                "nzu",
            )

        # pylint: disable=not-callable
        _ObserveZeroRule._kernel(
            grad,
            self.hyperparam.lr,
            self.hyperparam.momentum,
            param.data,
            self.state["v"],
            self.state["u"],
        )
        self.state["nzu"] = _ObserveZeroRule._nzu_kernel(grad, self.state["u"])


class LpMomentumSGD(optimizers.MomentumSGD):
    """ Inherits from MomentumSGD to use the low precision update rule. """

    def __init__(
        self,
        lr=_default_hyperparam.lr,
        momentum=_default_hyperparam.momentum,
        rule=None,
    ):
        super(LpMomentumSGD, self).__init__(lr=lr, momentum=momentum)

        # NOTE: you should be in float16 to use this optimizer.
        assert chainer.get_dtype() == np.float16
        # this will distinguish deterministic and stochastic rules.
        self._rule = rule

    def create_update_rule(self):
        if self._rule == "stochastic":
            return _StochasticRoundingRule(self.hyperparam)
        elif self._rule == "deterministic":
            return _DeterministicRoundUpRule(self.hyperparam)
        elif self._rule == "observe_zero":
            return _ObserveZeroRule(self.hyperparam)
        elif self._rule is None or self._rule == "origin":
            return MomentumSGDRule(self.hyperparam)
        else:
            raise ValueError("Cannot recognize LpMomentumRule: {}".format(self._rule))
