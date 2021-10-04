""" The linear link that supports adaptive loss scaling. """

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import utils

from chainerlp.functions.connection.ada_loss_linear import ada_loss_linear


class AdaLossLinear(L.Linear):
    """ Inherited from the Linear link in chainer. """

    def __init__(
        self,
        in_size,
        out_size=None,
        nobias=False,
        initialW=None,
        initial_bias=None,
        ada_loss_cfg=None,
        **kwargs
    ):
        """ """
        super().__init__(
            in_size,
            out_size=out_size,
            nobias=nobias,
            initialW=initialW,
            initial_bias=initial_bias,
        )

        # TODO: refactorize
        # To be passed to the ada loss function:
        if ada_loss_cfg is None:
            self.ada_loss_params = kwargs
        else:
            self.ada_loss_params = ada_loss_cfg

    def forward(self, x, n_batch_axes=1):
        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[1:])
            self._initialize_params(in_size)

        return ada_loss_linear(
            x, self.W, self.b, n_batch_axes=n_batch_axes, **self.ada_loss_params
        )
