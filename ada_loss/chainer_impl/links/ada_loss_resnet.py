""" Adaptive loss scaling supported Links for ResNet building blocks """

import chainer
import chainer.functions as F

from ada_loss.chainer_impl.links.identity_loss_scaling import IdentityLossScalingWrapper
from ada_loss.chainer_impl.functions.ada_loss_branch import AdaLossBranch
from ada_loss.chainer_impl.links.ada_loss_conv2d_bn_activ import AdaLossConv2DBNActiv
from ada_loss.chainer_impl.links.ada_loss_bn_activ_conv2d import AdaLossBNActivConv2D
from ada_loss.chainer_impl.links.ada_loss_convolution_2d import AdaLossConvolution2D


class AdaLossBasicBlock(chainer.Chain):
    """ Basic residual network block """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilate=1,
        groups=1,
        initialW=None,
        bn_kwargs={},
        residual=None,
        ada_loss_cfg=None,
    ):
        """ CTOR """
        super().__init__()

        self.ada_loss_cfg = ada_loss_cfg

        # parameters
        kwargs = {
            "ksize": 3,
            "pad": dilate,
            "nobias": True,
            "groups": groups,
            "initialW": initialW,
            "ada_loss_cfg": ada_loss_cfg,
            "bn_kwargs": bn_kwargs,
        }

        with self.init_scope():
            # pad = dilate
            self.conv1 = AdaLossConv2DBNActiv(
                in_channels, out_channels, stride=1, **kwargs
            )

            # parameters for the second conv
            kwargs["activ"] = None
            self.conv2 = AdaLossConv2DBNActiv(
                out_channels, out_channels, stride=stride, **kwargs
            )  # no ReLU after conv2

            # the additional mapping block on the residual connection
            if residual:
                self.residual = AdaLossConv2DBNActiv(
                    in_channels,
                    out_channels,
                    ksize=1,
                    stride=stride,
                    pad=0,
                    nobias=True,
                    initialW=initialW,
                    activ=None,
                    bn_kwargs=bn_kwargs,
                    ada_loss_cfg=ada_loss_cfg,
                )
            else:
                self.residual = lambda x: x

            self.relu = IdentityLossScalingWrapper(F.relu)

    def forward(self, x):
        x1, x2 = AdaLossBranch(ada_loss_cfg=self.ada_loss_cfg).apply((x,))
        h = self.conv1(x1)
        h = self.conv2(h)
        r = self.residual(x2)
        h = h + r  # seems to be unnecessary
        h = self.relu(h)

        return h


class AdaLossBottleneckBlock(chainer.Chain):
    """ Bottleneck residual network block, support AdaLoss """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        dilate=1,
        groups=1,
        initialW=None,
        bn_kwargs={},
        residual=False,
        ada_loss_cfg=None,
    ):
        """ CTOR """
        super().__init__()

        self.ada_loss_cfg = ada_loss_cfg

        # parameters will be shared by all
        kwargs = {
            "nobias": True,
            "initialW": initialW,
            "bn_kwargs": bn_kwargs,
            "ada_loss_cfg": ada_loss_cfg,
        }

        with self.init_scope():
            self.conv1 = AdaLossConv2DBNActiv(
                in_channels, mid_channels, ksize=1, pad=0, stride=stride, **kwargs
            )
            self.conv2 = AdaLossConv2DBNActiv(
                mid_channels,
                mid_channels,
                ksize=3,
                pad=dilate,
                dilate=dilate,
                groups=groups,
                stride=1,
                **kwargs
            )
            self.conv3 = AdaLossConv2DBNActiv(
                mid_channels,
                out_channels,
                ksize=1,
                stride=1,
                pad=0,
                activ=None,
                **kwargs
            )

            # the additional mapping block on the residual connection
            if residual:
                self.residual = AdaLossConv2DBNActiv(
                    in_channels,
                    out_channels,
                    ksize=1,
                    stride=stride,
                    pad=0,
                    activ=None,
                    **kwargs
                )
            else:
                self.residual = IdentityLossScalingWrapper(lambda x: x)

            self.relu = IdentityLossScalingWrapper(F.relu)

    def forward(self, x):
        x1, x2 = AdaLossBranch(ada_loss_cfg=self.ada_loss_cfg).apply((x,))
        h = self.conv1(x1)
        h = self.conv2(h)
        h = self.conv3(h)
        r = self.residual(x2)
        h = h + r
        h = self.relu(h)

        return h


class AdaLossBottleneckBlockv2(chainer.Chain):
    """ Bottleneck residual network block, support AdaLoss """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        dilate=1,
        groups=1,
        initialW=None,
        bn_kwargs={},
        residual=False,
        ada_loss_cfg=None,
    ):
        """ CTOR """
        super().__init__()

        self.ada_loss_cfg = ada_loss_cfg

        # parameters will be shared by all
        kwargs = {
            "nobias": True,
            "initialW": initialW,
            "bn_kwargs": bn_kwargs,
            "ada_loss_cfg": ada_loss_cfg,
        }

        with self.init_scope():
            self.conv1 = AdaLossBNActivConv2D(
                in_channels, mid_channels, ksize=1, pad=0, stride=stride, **kwargs
            )
            self.conv2 = AdaLossBNActivConv2D(
                mid_channels,
                mid_channels,
                ksize=3,
                pad=dilate,
                dilate=dilate,
                groups=groups,
                stride=1,
                **kwargs
            )
            self.conv3 = AdaLossBNActivConv2D(
                mid_channels,
                out_channels,
                ksize=1,
                stride=1,
                pad=0,
                activ=None,
                **kwargs
            )

            # the additional mapping block on the residual connection
            if residual:
                kwargs.pop("bn_kwargs")
                self.residual = AdaLossConvolution2D(
                    in_channels, out_channels, ksize=1, pad=0, stride=stride, **kwargs
                )
            else:
                self.residual = IdentityLossScalingWrapper(lambda x: x)

            self.relu = IdentityLossScalingWrapper(F.relu)

    def forward(self, x):
        x1, x2 = AdaLossBranch(ada_loss_cfg=self.ada_loss_cfg).apply((x,))
        h = self.conv1(x1)
        h = self.conv2(h)
        h = self.conv3(h)
        r = self.residual(x2)
        h = h + r
        # h = self.relu(h)

        return h
