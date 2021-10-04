import chainer
import chainer.links as L

from chainerlp.functions.connection.ada_loss_convolution_2d import (
    ada_loss_convolution_2d,
)


class AdaLossConvolution2D(L.Convolution2D):
    """ """

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=None,
        stride=1,
        pad=0,
        nobias=False,
        initialW=None,
        initial_bias=None,
        ada_loss_cfg=None,
        **kwargs
    ):
        super().__init__(
            in_channels,
            out_channels,
            ksize=ksize,
            stride=stride,
            pad=pad,
            nobias=nobias,
            initialW=initialW,
            initial_bias=initial_bias,
            **kwargs
        )

        self.ada_loss_cfg = {} if ada_loss_cfg is None else ada_loss_cfg

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        return ada_loss_convolution_2d(
            x,
            self.W,
            self.b,
            self.stride,
            self.pad,
            ada_loss_cfg=self.ada_loss_cfg,
            dilate=self.dilate,
            groups=self.groups,
        )
