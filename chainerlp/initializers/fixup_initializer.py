""" Implement fixup initializer. """

import numpy as np

import chainer
import chainer.initializers as I
from chainer import backend
from chainer.backends import cuda
from chainer import initializer


class FixupNormal(initializer.Initializer):
    """ The normal initializer proposed by the Fixup paper. """

    def __init__(self, L, m, dtype=None, fan_option="fan_out"):
        """ CTOR """
        super(FixupNormal, self).__init__(dtype)

        self.L = L
        self.m = m
        self.fan_option = fan_option

    def __call__(self, array):
        """ The initializer. """
        scale = self.L ** (-1 / (2 * self.m - 2))
        I.HeNormal(scale=scale, dtype=self.dtype, fan_option=self.fan_option)(array)
