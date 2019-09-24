""" A trainer extension that monitors the numerical issues
    that may happen during training in low-precision data types.
"""


from chainer.training import extension


class NumericalDebugger(extension.Extension):
    """ Detect the numerical issues from the parameters that the given
        optimizer is working on.

        Designed based on the `FailOnNonNumber` extension.
    """

    def __call__(self, trainer):
        optimizers = trainer.updater.get_all_optimizers()

        for name, optimizer in optimizers.items():
            target = optimizer.target  # target link object
            xp = target.xp  # the array module, could be numpy, cupy, or chainerx

            is_diverged = False
            for param_name, param in sorted(target.namedparams(), key=lambda x: x[0]):
                if not xp.isfinite(param.array).all():
                    is_diverged = True
                    # print('==> Parameter {} in optimizer \'{}\' diverge'.format(
                    #     param_name, name))
                    # print(param)

            if is_diverged:
                raise RuntimeError(
                    'Kill the process since parameters in optimizer'
                    ' \'{}\' diverge at step {}. R.I.P.'.format(name, optimizer.t))
