""" Parameter specific learning rate shift extension """

import numpy

import chainer
from chainer.training import extension
import chainer.training.extensions as E


class ExponentialShift(E.ExponentialShift):
    """ Shift the learning rate by setting up the lr field of
        parameters' hyperparam. """

    def __init__(
        self,
        attr,
        rate,
        params_config=None,
        schedule=None,
        warmup_attr_ratio=None,
        warmup_n_epoch=None,
        init=None,
        target=None,
        optimizer=None,
    ):
        """ CTOR.
            `param_config` list the `Parameter`s that should be treated differently.
        """
        # The default CTOR will update optimizer
        super().__init__(attr, rate, init=init, target=target, optimizer=optimizer)
        self._iter = 0
        self.schedule = schedule
        self.params_config = [] if params_config is None else params_config
        self.warmup_attr_ratio = warmup_attr_ratio
        self.warmup_n_epoch = warmup_n_epoch

    def __call__(self, trainer):
        """ When the shift is requested """
        self._iter += 1

        if self.warmup_n_epoch is not None and self._iter == self.warmup_n_epoch:
            # deal with normal parameters
            value = self._init / self.warmup_attr_ratio  # warmup value
            self._init = value
            optimizer = self._get_optimizer(trainer)
            setattr(optimizer, self._attr, value)

            # TODO: refactorize
            # load different configurations
            for config in self.params_config:
                params, init = config["params"], config["init"]
                value = init / self.warmup_attr_ratio
                config["init"] = value

                for param in params:
                    assert isinstance(param, chainer.Parameter)
                    setattr(param.update_rule.hyperparam, self._attr, value)
        elif self.schedule is not None and self._iter in self.schedule:
            # Update the field in the optimizer correspondingly
            super().__call__(trainer)

            # load different configurations
            for config in self.params_config:
                params, init = config["params"], config["init"]
                value = init * (self._rate ** self._t)

                for param in params:
                    assert isinstance(param, chainer.Parameter)
                    setattr(param.update_rule.hyperparam, self._attr, value)
