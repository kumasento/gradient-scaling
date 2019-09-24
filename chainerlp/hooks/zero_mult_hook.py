""" Hook to sample the zero multiplications """

import os
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import backend
from chainer import link_hook
from chainer import function_hook
from chainer.training import Trainer


class ZeroMultFuncHook(function_hook.FunctionHook):
    """ The zero-mult sampling function hook. """

    name = 'ZeroMultFuncHook'

    def __init__(self,
                 trainer=None,
                 sample_per_n_iteration=100,
                 snapshot_dir=None):
        """ CTOR. """
        assert isinstance(trainer, Trainer)

        self.trainer = trainer
        self.sample_per_n_iteration = sample_per_n_iteration
        self.results = []  # record the final result
        self.counters = OrderedDict()
        self.n_iter = 0
        self.snapshot_dir = snapshot_dir

    @property
    def optim(self):
        optims = self.trainer.updater.get_all_optimizers()
        return optims['main']

    @property
    def current_iteration(self):
        return self.trainer.updater.iteration

    @property
    def current_epoch(self):
        return self.optim.epoch

    def inc_counter(self, func):
        """ Increase the counter of the current label func """
        if func.label not in self.counters:
            self.counters[func.label] = 0
        else:
            self.counters[func.label] += 1
        return self.counters[func.label]

    def reset_counter(self):
        self.counters.clear()

    def forward_preprocess(self, func, in_data):
        """ called before running the forward pass """
        if self.current_iteration % self.sample_per_n_iteration != 1:
            return None

        if self.n_iter != self.current_iteration:  # reset the counter
            self.n_iter = self.current_iteration
            self.reset_counter()

        if func.label in ['Convolution2DFunction']:
            self.process_convolution2d(func, in_data)

    def process_convolution2d(self, func, in_data):
        """ Work on the convolution2d input. """
        assert len(in_data) == 2

        func_id = self.inc_counter(func)
        X, W = in_data
        xp = backend.get_array_module(X)

        ksize = W.shape[2]
        stride = 1
        pad = 1 if ksize == 3 else 0

        X_ = F.im2col(X, ksize, stride=stride, pad=pad).reshape(
            [X.shape[0], -1, X.shape[2] * X.shape[3]])
        X_ = F.transpose(X_, axes=(0, 2, 1)).reshape([-1, X_.shape[1]])
        X_ = X_.array

        W_ = W.reshape([W.shape[0], -1])

        W_nz = (W_ != 0).astype('bool')
        X_nz = (X_ != 0).astype('bool')

        n_zm = 0
        for i in range(W_.shape[0]):
            M = xp.multiply(X_, W_[i, :])  # multiply

            # zero mult
            ZM = xp.logical_and(M == 0, xp.logical_and(W_nz[i, :], X_nz))
            n_zm += ZM.sum()

        self.results.append([
            self.current_epoch, self.current_iteration, func_id, func.label,
            n_zm.item(), W_.shape[0] * X_.shape[0] * X_.shape[1]
        ])

    def snapshot(self):
        """ Take a snapshot of zero mult results """
        fp = os.path.join(self.snapshot_dir, 'zero_mult.csv')
        df = pd.DataFrame(
            self.results,
            columns=['n_epoch', 'n_iter', 'func_id', 'func_label', 'n_zm', 'n_total'])
        df.to_csv(fp)