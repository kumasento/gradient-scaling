""" Snapshot the gradient """
import chainer
from chainer.serializers import npz
from chainer.utils import argument
from chainer.training import extensions
from chainer.training.extensions import snapshot_writers


class GradientSnapshot(extensions._snapshot._Snapshot):
    """ According to Slack discussion. """

    def __init__(
        self,
        target=None,
        condition=None,
        writer=None,
        filename="snapshot_iter_{.updater.iteration}",
        snapshot_on_error=False,
        model=None,
    ):
        super(GradientSnapshot, self).__init__(
            target=target,
            condition=condition,
            writer=writer,
            filename=filename,
            snapshot_on_error=snapshot_on_error,
        )

        assert isinstance(model, chainer.Chain)
        self._model = model

    def _make_snapshot(self, trainer):
        """ only override this function """
        # target = trainer if self._target is None else self._target

        target = {}
        for name, p in self._model.namedparams():
            target[name] = chainer.cuda.to_cpu(p.data)
            target["{}_grad".format(name)] = chainer.cuda.to_cpu(p.grad)

        # serialized_target = npz.serialize(target)
        serialized_target = target

        filename = self.filename
        if callable(filename):
            filename = filename(trainer)
        else:
            filename = filename.format(trainer)
        outdir = trainer.out
        self.writer(filename, outdir, serialized_target)


def gradient_snapshot(
    savefun=None, filename="snapshot_iter_{.updater.iteration}", **kwargs
):
    target, condition, writer, snapshot_on_error, model = argument.parse_kwargs(
        kwargs,
        ("target", None),
        ("condition", None),
        ("writer", None),
        ("snapshot_on_error", False),
        ("model", None),
    )
    argument.assert_kwargs_empty(kwargs)

    if savefun is not None and writer is not None:
        raise TypeError("savefun and writer arguments cannot be specified together.")

    if writer is None:
        if savefun is None:
            savefun = npz.save_npz
        writer = snapshot_writers.SimpleWriter(savefun=savefun)

    return GradientSnapshot(
        target=target,
        condition=condition,
        writer=writer,
        filename=filename,
        snapshot_on_error=snapshot_on_error,
        model=model,
    )
