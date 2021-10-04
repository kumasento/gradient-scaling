""" Almost a replicate of Conv2DBNActiv in chainercv.links.connection. """

import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass

from ada_loss.chainer_impl.links.ada_loss_convolution_2d import AdaLossConvolution2D
from ada_loss.chainer_impl.links.ada_loss_batch_normalization import (
    AdaLossBatchNormalization,
)
from ada_loss.chainer_impl.links.identity_loss_scaling import IdentityLossScalingWrapper


class AdaLossBNActivConv2D(chainer.Chain):
    """ A Conv2DBNActiv that allow you to use custom BN function. """

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=None,
        stride=1,
        pad=0,
        dilate=1,
        groups=1,
        nobias=True,
        initialW=None,
        initial_bias=None,
        activ=relu,
        use_bn=True,
        bn_kwargs={},
        ada_loss_cfg=None,
    ):
        super().__init__()
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        if activ is not None:
            self.activ = IdentityLossScalingWrapper(activ)
        else:
            self.activ = None

        with self.init_scope():
            self.conv = AdaLossConvolution2D(
                in_channels,
                out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=nobias,
                initialW=initialW,
                initial_bias=initial_bias,
                dilate=dilate,
                groups=groups,
                ada_loss_cfg=ada_loss_cfg,
            )

            # TODO: allow passing customized BN
            if use_bn:
                if "comm" in bn_kwargs:
                    raise ValueError(
                        "comm is not supported for AdaLossBatchNormalization"
                    )
                    # self.bn = MultiNodeBatchNormalization(
                    #     out_channels, **bn_kwargs)
                else:
                    self.bn = AdaLossBatchNormalization(
                        in_channels, ada_loss_cfg=ada_loss_cfg, **bn_kwargs
                    )
            else:
                self.bn = None

    def forward(self, x):
        if self.bn is not None:
            h = self.bn(x)
        if self.activ is not None:
            h = self.activ(h)
        h = self.conv(h)
        return h
