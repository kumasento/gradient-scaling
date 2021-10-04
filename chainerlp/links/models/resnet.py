""" Implements ResNet modules for CIFAR-10/100.

Changelog:
(06/07) add support for the fixup initialization method based on
  https://github.com/hongyi-zhang/Fixup/blob/master/cifar/models/fixup_resnet_cifar.py
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer import backend
from chainer import static_graph
from chainer import initializers

from chainercv.links import PickableSequentialChain
from chainercv.links import SEBlock

from chainerlp.links import Conv2DBNActiv, FixupConv2D, FixupIdentity
from chainerlp.initializers.fixup_initializer import FixupNormal


class BasicBlock(chainer.Chain):
    """ A basic residule block used in ResNet modules for CIFAR. 
        BasicBlock naming is inherited from the `pytorch-classification` repository.

        Based on the implementation in `chainercv.links.model.resnet.resblock.ResBlock`.

        `res_scale` is from the Inception-ResNet paper
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilate=1,
        groups=1,
        initialW=None,
        bn_kwargs={},
        residual_conv=False,  # TODO: remove this
        stride_first=False,
        add_seblock=False,
        res_scale=None,
        use_fixup=False,
    ):
        """ CTOR """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilate = dilate
        self.groups = groups
        self.initialW = initialW
        self.bn_kwargs = bn_kwargs
        self.residual_conv = residual_conv

        if res_scale is not None:
            if use_fixup:
                raise ValueError("Cannot use fixup when res_scale is not None")
            self.res_scale = res_scale
        else:
            self.res_scale = 1.0

        # Fixup
        self.use_fixup = use_fixup
        self.cached_zeros = None

        # Conv function
        ConvLink = Conv2DBNActiv if not use_fixup else FixupConv2D
        # parameters
        kwargs = {
            "ksize": 3,
            "pad": dilate,
            "nobias": True,
            "groups": groups,
            "initialW": initialW,
        }
        if not use_fixup:
            kwargs["bn_kwargs"] = bn_kwargs
        else:
            kwargs["use_scale"] = False

        with self.init_scope():
            # pad = dilate
            self.conv1 = ConvLink(in_channels, out_channels, stride=1, **kwargs)

            # parameters for the second conv
            kwargs["activ"] = None
            if use_fixup:
                kwargs["initialW"] = None
                kwargs["use_scale"] = True  # turn on use scale

            self.conv2 = ConvLink(
                out_channels, out_channels, stride=stride, **kwargs
            )  # no ReLU after conv2

            # Squeeze-and-Excitation
            if add_seblock:
                # TODO: check whether this block will affect the numerical stability of a model
                self.se = SEBlock(out_channels)

            # the additional mapping block on the residual connection
            if residual_conv:
                if not use_fixup:
                    self.residual = Conv2DBNActiv(
                        in_channels,
                        out_channels,
                        ksize=1,
                        stride=stride,
                        pad=0,
                        nobias=True,
                        initialW=initialW,
                        activ=None,
                        bn_kwargs=bn_kwargs,
                    )
                else:  # When using fixup, we pass the residual connection through average pooling
                    self.residual = FixupIdentity(stride)
            else:
                self.residual = lambda x: x

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        if self.use_fixup:
            x += self.conv1.bias_in
        residual = self.residual(x)

        h = h + self.res_scale * residual

        h = F.relu(h)

        return h


class Bottleneck(chainer.Chain):
    """ Replica of chainer.cv.links.model.resnet.resblock 

    We need to do this to replace the BN function.
    """

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
    ):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilate = dilate
        self.groups = groups
        self.initialW = initialW
        self.bn_kwargs = bn_kwargs
        self.residual_conv = residual

        with self.init_scope():
            self.conv1 = Conv2DBNActiv(
                in_channels,
                mid_channels,
                ksize=1,
                stride=stride,
                pad=0,
                nobias=True,
                initialW=initialW,
                bn_kwargs=bn_kwargs,
            )
            self.conv2 = Conv2DBNActiv(
                mid_channels,
                mid_channels,
                ksize=3,
                stride=1,
                pad=dilate,
                dilate=dilate,
                groups=groups,
                nobias=True,
                initialW=initialW,
                bn_kwargs=bn_kwargs,
            )
            self.conv3 = Conv2DBNActiv(
                mid_channels,
                out_channels,
                ksize=1,
                stride=1,
                pad=0,
                nobias=True,
                initialW=initialW,
                activ=None,
                bn_kwargs=bn_kwargs,
            )

            if residual:
                self.residual = Conv2DBNActiv(
                    in_channels,
                    out_channels,
                    ksize=1,
                    stride=stride,
                    pad=0,
                    nobias=True,
                    initialW=initialW,
                    activ=None,
                    bn_kwargs=bn_kwargs,
                )
            else:
                self.residual = lambda x: x

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        r = self.residual(x)
        h += r
        h = F.relu(h)

        return h


class ResBasicBlock(PickableSequentialChain):
    """ Basic building block for ResNet.

    Should consider differently for ResNet-50/101/152 and ResNet-110.
    """

    def __init__(
        self,
        n_layer,
        in_channels,
        out_channels,
        stride,
        dilate=1,
        groups=1,
        initialW=None,
        bn_kwargs={},
        stride_first=False,
        add_seblock=False,
        use_fixup=False,
        res_scale=None,
    ):
        super(ResBasicBlock, self).__init__()

        with self.init_scope():
            # Follows the naming convention
            self.a = BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                dilate=dilate,
                groups=groups,
                initialW=initialW,
                bn_kwargs=bn_kwargs,
                residual_conv=True,
                stride_first=stride_first,
                add_seblock=add_seblock,
                use_fixup=use_fixup,
                res_scale=res_scale,
            )

            for i in range(n_layer - 1):
                name = "b{}".format(i)
                block = BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dilate=dilate,
                    initialW=initialW,
                    bn_kwargs=bn_kwargs,
                    residual_conv=False,
                    add_seblock=add_seblock,
                    groups=groups,
                    use_fixup=use_fixup,
                    res_scale=res_scale,
                )
                setattr(self, name, block)


class ResBottleneckBlock(PickableSequentialChain):
    """ Residual block using bottleneck """

    def __init__(
        self,
        n_layer,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilate=1,
        groups=1,
        initialW=None,
        bn_kwargs={},
    ):
        super(ResBottleneckBlock, self).__init__()
        # Dilate option is applied to all bottlenecks.
        with self.init_scope():
            self.a = Bottleneck(
                in_channels,
                mid_channels,
                out_channels,
                stride,
                dilate,
                groups,
                initialW,
                bn_kwargs=bn_kwargs,
                residual=True,
            )
            for i in range(n_layer - 1):
                name = "b{}".format(i + 1)
                bottleneck = Bottleneck(
                    out_channels,
                    mid_channels,
                    out_channels,
                    stride=1,
                    dilate=dilate,
                    initialW=initialW,
                    bn_kwargs=bn_kwargs,
                    residual=False,
                    groups=groups,
                )
                setattr(self, name, bottleneck)


class ResNetCIFAR(PickableSequentialChain):
    """ A basic class for a complete ResNet model, designed for CIFAR. """

    _blocks = {
        20: [3, 3, 3],
        32: [5, 5, 5],
        44: [7, 7, 7],
        56: [9, 9, 9],
        110: [18, 18, 18],
        1202: [200, 200, 200],
    }

    def __init__(
        self,
        n_layer,
        n_class=None,
        initialW=None,
        fc_kwargs={},
        use_fixup=False,
        res_scale=None,
    ):
        """ CTOR. """
        super().__init__()

        stride_first = False
        blocks = self._blocks[n_layer]

        # configure initializers
        if initialW is None:
            initialW = self.get_initialW(
                use_fixup=use_fixup, L=sum(self._blocks[n_layer]), m=2
            )
        if "initialW" not in fc_kwargs:
            if use_fixup:
                fc_kwargs["initialW"] = initializers.Zero()
            else:
                fc_kwargs["initialW"] = initialW

        kwargs = {
            "initialW": initialW,
            "stride_first": stride_first,
            "use_fixup": use_fixup,
        }

        # TODO: consider different scale arrangements
        res_scales = [res_scale] * 3

        with self.init_scope():
            # bias parameters for the first CONV and the last FC
            if use_fixup:
                self.bias1 = chainer.Parameter(initializer=initializers.Zero(), shape=1)
                self.bias2 = chainer.Parameter(initializer=initializers.Zero(), shape=1)

                # first layer is connected with BN and ReLU
                self.first_bias_add = lambda x: x + self.bias1

            self.conv1 = Conv2DBNActiv(
                3,
                16,
                ksize=3,
                pad=1,
                nobias=True,
                initialW=self.get_initialW(use_fixup=False),
            )
            # NOTE: no pooling connected for CIFAR ResNet model
            self.res2 = ResBasicBlock(
                blocks[0], 16, 16, stride=1, res_scale=res_scales[0], **kwargs
            )
            self.res3 = ResBasicBlock(
                blocks[1], 16, 32, stride=2, res_scale=res_scales[1], **kwargs
            )
            self.res4 = ResBasicBlock(
                blocks[2], 32, 64, stride=2, res_scale=res_scales[2], **kwargs
            )
            self.pool5 = lambda x: F.average_pooling_2d(x, ksize=(8, 8))
            if use_fixup:
                self.last_bias_add = lambda x: x + self.bias2
            # NOTE: it should be explicit
            self.squeeze = lambda x: F.squeeze(x, axis=(2, 3))
            self.fc6 = L.Linear(64, n_class, **fc_kwargs)

    def get_initialW(self, use_fixup=False, L=None, m=None):
        if not use_fixup:
            return initializers.HeNormal(scale=1.0, fan_option="fan_out")
        else:
            return FixupNormal(L, m)


class ResNet(PickableSequentialChain):
    """ ResNet models contructed for training on ImageNet. """

    _blocks = {
        18: [2, 2, 2, 2],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }

    def __init__(
        self, n_layer, n_class=None, initialW=None, fc_kwargs={}, first_bn_mixed16=False
    ):
        """ CTOR. """
        super(ResNet, self).__init__()

        conv1_no_bias = n_layer != 50
        blocks = self._blocks[n_layer]

        if initialW is None:
            initialW = initializers.HeNormal(scale=1.0, fan_option="fan_out")
        if "initialW" not in fc_kwargs:
            fc_kwargs["initialW"] = initializers.Normal(scale=0.01)
        kwargs = {"initialW": initialW}

        with self.init_scope():
            bn_kwargs = {} if not first_bn_mixed16 else {"dtype": chainer.mixed16}
            self.conv1 = Conv2DBNActiv(
                3,
                64,
                ksize=7,
                stride=2,
                pad=3,
                nobias=conv1_no_bias,
                initialW=initialW,
                bn_kwargs=bn_kwargs,
            )
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)

            # TODO: refactorize this part
            if n_layer < 50:
                last_n_channel = 512
                self.res2 = ResBasicBlock(blocks[0], 64, 64, stride=1, **kwargs)
                self.res3 = ResBasicBlock(blocks[1], 64, 128, stride=2, **kwargs)
                self.res4 = ResBasicBlock(blocks[2], 128, 256, stride=2, **kwargs)
                self.res5 = ResBasicBlock(
                    blocks[3], 256, last_n_channel, stride=2, **kwargs
                )
            else:
                last_n_channel = 2048
                self.res2 = ResBottleneckBlock(
                    blocks[0], 64, 64, 256, stride=1, **kwargs
                )
                self.res3 = ResBottleneckBlock(
                    blocks[1], 256, 128, 512, stride=2, **kwargs
                )
                self.res4 = ResBottleneckBlock(
                    blocks[2], 512, 256, 1024, stride=2, **kwargs
                )
                self.res5 = ResBottleneckBlock(
                    blocks[3], 1024, 512, last_n_channel, stride=2, **kwargs
                )
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(last_n_channel, n_class, nobias=False, **fc_kwargs)


class resnet20(ResNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(resnet20, self).__init__(20, n_class=n_class, **kwargs)


class resnet32(ResNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(resnet32, self).__init__(32, n_class=n_class, **kwargs)


class resnet44(ResNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(resnet44, self).__init__(44, n_class=n_class, **kwargs)


class resnet56(ResNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(resnet56, self).__init__(56, n_class=n_class, **kwargs)


class resnet110(ResNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(resnet110, self).__init__(110, n_class=n_class, **kwargs)


class resnet1202(ResNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(resnet1202, self).__init__(1202, n_class=n_class, **kwargs)


# ImageNet models


class resnet18(ResNet):
    def __init__(self, n_class=None, **kwargs):
        super(resnet18, self).__init__(18, n_class=n_class, **kwargs)


class resnet50(ResNet):
    def __init__(self, n_class=None, **kwargs):
        super(resnet50, self).__init__(50, n_class=n_class, **kwargs)


class resnet101(ResNet):
    def __init__(self, n_class=None, **kwargs):
        super(resnet101, self).__init__(101, n_class=n_class, **kwargs)


class resnet152(ResNet):
    def __init__(self, n_class=None, **kwargs):
        super(resnet152, self).__init__(152, n_class=n_class, **kwargs)
