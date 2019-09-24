""" This extension implements the LR schedule from the SGDR paper.
Reference:
[1] https://gist.github.com/hrsma2i/9c6514e94cd5e802d9e216aef2bcfe59
"""

import numpy as np
from chainer.training import extension


class RestartLRScheduler(extension.Extension):
    """ Implements the extension """

    def __init__(self,
                 lr_max,
                 lr_min,
                 T_0,
                 T_mult,
                 max_n_mult=10,
                 optimizer=None):
        """ CTOR """
        super(RestartLRScheduler, self).__init__()

        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self.max_n_mult = max_n_mult
        self.optimizer = optimizer

        # state
        self._t = 0
        self._last_lr = None
        self._last_t_i = None
        self._last_T_i = None  # to calculate the current T

    def initialize(self, trainer):
        """ Initialize the content in the trainer """
        optimizer = self._get_optimizer(trainer)

        if self._last_lr is not None:  # resuming from snapshot
            self._update_value(optimizer, 'lr', self._last_lr)
            self._update_value(optimizer, 't_i', self._last_t_i)
            self._update_value(optimizer, 'T_i', self._last_T_i)
        else:
            self._update_value(optimizer, 'lr', self.lr_max)
            self._update_value(optimizer, 't_i', 1)  # starting from 1
            self._update_value(optimizer, 'T_i', self.T_0)

    def serialize(self, serializer):
        """ Resume """
        self._t = serializer('_t', self._t)
        self._last_lr = serializer('_last_lr', self._last_lr)
        self._last_t_i = serializer('_last_t_i', self._last_t_i)
        self._last_T_i = serializer('_last_T_i', self._last_T_i)

    def __call__(self, trainer):
        """ Main update function. """
        self._t += 1  # starting from 1

        optimizer = self._get_optimizer(trainer)

        # period is a range between applying T_mult
        i = self._get_current_period()

        T_i = self.T_0 * (self.T_mult**i)
        # apply geometric series formula
        if self.T_mult != 1:
            t_i = self._t - (T_i - self.T_0) // (self.T_mult - 1) + 1
        else:
            t_i = self._t - T_i * i + 1

        # collect learning rate
        lr = self._get_lr(T_i, t_i)

        self._update_value(optimizer, 'lr', lr)
        self._update_value(optimizer, 't_i', t_i)
        self._update_value(optimizer, 'T_i', T_i)

    def _get_lr(self, T_i, t_i):
        """ Get the learning rate at period p """
        # current learning rate
        return self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
            1 + np.cos(np.pi * (t_i - 1) / T_i))

    def _get_current_period(self):
        """ Get the T_cur value """
        periods = [self.T_0 * (self.T_mult**i) for i in range(self.max_n_mult)]
        cumsum_periods = np.cumsum(periods)

        return np.where((self._t - cumsum_periods) < 0)[0][0]

    def _get_optimizer(self, trainer):
        """ Collect optimizer """
        return self.optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, attr, value):
        """ Update the attr in optimizer """
        setattr(optimizer, attr, value)
        setattr(self, '_last_{}'.format(attr), value)  # naming convention
