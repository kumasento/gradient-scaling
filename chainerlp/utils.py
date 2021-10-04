""" Utility """

import os
import random
import numpy as np
import cupy as cp
import chainer


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


def get_model_size(model, excludes=None):
    """ Get the number of parameters of a given model """
    assert isinstance(model, chainer.Chain)
    if excludes is None:
        excludes = []

    n_params = 0
    for param in model.params():
        if param.name in excludes:
            continue
        if param.array is None:  # scala parameters
            print(type(model), param.name, type(param))
            continue
        n_params += param.size

    return n_params
