""" Transform links defined in ChainerCV. """

from ada_loss.chainer_impl.transforms.base import AdaLossTransform
from ada_loss.chainer_impl.links import AdaLossConv2DBNActiv
from ada_loss.chainer_impl.links import AdaLossBottleneckBlock

import chainer.links as L
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet.resblock import Bottleneck


class AdaLossTransformConv2DBNActiv(AdaLossTransform):
    """ Transform Conv2DBNActiv defined in chainercv.links """

    cls = Conv2DBNActiv

    def create(self, link, cfg):
        """ """
        assert isinstance(link.bn, L.BatchNormalization)
        return AdaLossConv2DBNActiv(
            link.conv.in_channels,
            link.conv.out_channels,
            ksize=link.conv.ksize,
            stride=link.conv.stride,
            pad=link.conv.pad,
            dilate=link.conv.dilate,
            groups=link.conv.groups,
            nobias=link.conv.b is None,
            ada_loss_cfg=cfg,
        )

    def copyparams(self, src, dst):
        """ """
        dst.conv.copyparams(src.conv)
        dst.bn.copyparams(src.bn)


class AdaLossTransformBottleneck(AdaLossTransform):
    """ """

    cls = Bottleneck

    def create(self, link, cfg):
        """ """
        assert not hasattr(link, "se")
        assert isinstance(link.conv1.bn, L.BatchNormalization)

        return AdaLossBottleneckBlock(
            link.conv1.conv.in_channels,
            link.conv2.conv.in_channels,
            link.conv3.conv.out_channels,
            stride=link.conv1.conv.stride,
            dilate=link.conv2.conv.dilate,
            groups=link.conv2.conv.groups,
            residual=hasattr(link, "residual_conv"),
            ada_loss_cfg=cfg,
        )

    def copyparams(self, src, dst):
        """ """
        dst.conv1.copyparams(src.conv1)
        dst.conv2.copyparams(src.conv2)
        dst.conv3.copyparams(src.conv3)

        if hasattr(src, "residual_conv"):
            dst.residual.copyparams(src.residual_conv)
