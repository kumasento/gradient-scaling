""" Utility functions """

import os
import random
import numpy as np
import cupy as cp
import chainer


def scale_grad(grad, scale, dtype=None, key="loss_scale"):
    """ Scale the gradient value by scale and attach 'loss_scale' """
    assert isinstance(grad, chainer.Variable), "grad should be a variable"

    # compute the scaled gradient
    xp = chainer.backend.get_array_module(grad.array)
    if dtype is None:
        dtype = chainer.get_dtype()
    scale = xp.array(scale, dtype=dtype)
    grad.data *= scale

    # attach the loss scaling key
    grad.__dict__[key] = scale

    return grad


def set_random_seed(seed, device=None):
    """ Set random seed before running training.

    Refs:
        https://qiita.com/TokyoMickey/items/cc8cd43545f2656b1cbd
        https://github.com/chainer/chainer/issues/4550
    """
    print(
        "==> Set manual random seed to {} in process PID={} PPID={} on device={}".format(
            seed, os.getpid(), os.getppid(), device
        )
    )

    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)

    # NOTE: use device before setting up the random generator
    # https://github.com/chainer/chainer/issues/4487
    if device is not None:
        chainer.backends.cuda.get_device_from_id(int(device)).use()

    # set Chainer(CuPy) random seed
    cp.random.seed(seed)
    # force cuDNN to be deterministic
    chainer.global_config.cudnn_deterministic = True
