""" Adaptive loss scaling related experiments """
import chainer
from chainerlp import train_utils, utils
from chainerlp.hooks.ada_loss_monitor import AdaLossMonitor

from ada_loss.chainer_impl import *


def compare_loss_scaling_by_nonzero(
    net,
    cfgs,
    init_scale=1,
    dataset="mnist",
    verbose=False,
    sample_iterations=None,
    includes=None,
    manual_seed=0,
    device=-1,
    dtype=chainer.mixed16,
    n_epoch=10,
    learnrate=0.01,
):
    """ Collect the number of nonzero at various points during
        training and see their relationship with the loss scaling
        method.

        The model should be provided.

        Different loss scaling method is specified by cfgs (a list)
    """
    # history
    hists = []

    with chainer.using_config("dtype", dtype):
        # create loss scaled model
        for i, cfg in enumerate(cfgs):
            net_ = net.copy(mode="copy")  # deeply copy the original link
            # if init_scales:
            net_ = AdaLossScaled(net_, init_scale=init_scale, cfg=cfg, verbose=verbose)
            # prepare the hook that records necessary values
            hook = AdaLossMonitor(
                sample_iterations=sample_iterations, includes=includes, verbose=verbose
            )
            # collect data
            utils.set_random_seed(manual_seed, device=device)
            if dataset == "mnist":
                hooks, log = train_utils.train_model_on_mnist(
                    net_,
                    epoch=n_epoch,
                    batchsize=128,
                    device=device,
                    learnrate=learnrate,
                    hooks=[hook],
                )
            else:
                raise ValueError("dataset name not found: {}".format(dataset))

    # prepare the sampled results
    df = hooks[0].export_history()
