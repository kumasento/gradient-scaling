import chainer
import chainer.links as L

from chainerlp.links import Conv2DBNActiv
from chainerlp.links import BNActivConv2D 
from chainerlp.links.models.resnet import BasicBlock, Bottleneck
from chainerlp.links.models.resnet_v2 import _Bottleneck as Bottleneckv2

from ada_loss.chainer_impl.links import *
from ada_loss.chainer_impl.ada_loss_transforms import AdaLossTransform


class AdaLossTransformBasicBlock(AdaLossTransform):
    cls = BasicBlock

    def __call__(self, link, cfg=None):
        """ """
        assert isinstance(link, self.cls)
        # create new link
        link_ = AdaLossBasicBlock(link.in_channels,
                                  link.out_channels,
                                  stride=link.stride,
                                  dilate=link.dilate,
                                  groups=link.groups,
                                  initialW=link.initialW,
                                  bn_kwargs=link.bn_kwargs,
                                  residual=link.residual_conv,
                                  ada_loss_cfg=cfg)

        # copy parameters
        link_.conv1.copyparams(link.conv1)
        link_.conv2.copyparams(link.conv2)
        if link.residual_conv:
            link_.residual.copyparams(link.residual)

        return link_


class AdaLossTransformBottleneck(AdaLossTransform):
    """  """
    cls = Bottleneck

    def __call__(self, link, cfg=None):
        """ """
        assert isinstance(link, self.cls)
        # create new link
        link_ = AdaLossBottleneckBlock(link.in_channels,
                                       link.mid_channels,
                                       link.out_channels,
                                       stride=link.stride,
                                       dilate=link.dilate,
                                       groups=link.groups,
                                       initialW=link.initialW,
                                       bn_kwargs=link.bn_kwargs,
                                       residual=link.residual_conv,
                                       ada_loss_cfg=cfg)

        # copy parameters
        link_.conv1.copyparams(link.conv1)
        link_.conv2.copyparams(link.conv2)
        link_.conv3.copyparams(link.conv3)
        if link.residual_conv:
            link_.residual.copyparams(link.residual)

        return link_

class AdaLossTransformBottleneckv2(AdaLossTransform):
    """  """
    cls = Bottleneckv2

    def __call__(self, link, cfg=None):
        """ """
        assert isinstance(link, self.cls)
        # create new link
        link_ = AdaLossBottleneckBlockv2(link.in_channels,
                                       link.mid_channels,
                                       link.out_channels,
                                       stride=link.stride,
                                       dilate=link.dilate,
                                       groups=link.groups,
                                       initialW=link.initialW,
                                       bn_kwargs=link.bn_kwargs,
                                       residual=link.residual_conv,
                                       ada_loss_cfg=cfg)

        # copy parameters
        link_.conv1.copyparams(link.conv1)
        link_.conv2.copyparams(link.conv2)
        link_.conv3.copyparams(link.conv3)
        if link.residual_conv:
            link_.residual.copyparams(link.residual)

        return link_

class AdaLossTransformConv2DBNActiv(AdaLossTransform):
    """ """
    cls = Conv2DBNActiv

    def __call__(self, link, cfg=None):
        """ """
        assert isinstance(link, self.cls)

        # create new link
        link_ = AdaLossConv2DBNActiv(link.in_channels,
                                     link.out_channels,
                                     ksize=link.ksize,
                                     stride=link.stride,
                                     pad=link.pad,
                                     dilate=link.dilate,
                                     groups=link.groups,
                                     nobias=link.nobias,
                                     initialW=link.initialW,
                                     initial_bias=link.initial_bias,
                                     use_bn=link.use_bn,
                                     bn_kwargs=link.bn_kwargs,
                                     ada_loss_cfg=cfg)

        # copy parameters
        link_.copyparams(link)

        return link_

class AdaLossTransformBNActivConv2D(AdaLossTransform):
    """ """
    cls = BNActivConv2D

    def __call__(self, link, cfg=None):
        """ """
        assert isinstance(link, self.cls)

        # create new link
        link_ = AdaLossBNActivConv2D(link.in_channels,
                                     link.out_channels,
                                     ksize=link.ksize,
                                     stride=link.stride,
                                     pad=link.pad,
                                     dilate=link.dilate,
                                     groups=link.groups,
                                     nobias=link.nobias,
                                     initialW=link.initialW,
                                     initial_bias=link.initial_bias,
                                     use_bn=link.use_bn,
                                     bn_kwargs=link.bn_kwargs,
                                     ada_loss_cfg=cfg)

        # copy parameters
        link_.copyparams(link)

        return link_
