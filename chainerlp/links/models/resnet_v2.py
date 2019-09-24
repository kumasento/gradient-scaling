""" This repository implements the ResNet variants with pre-activation.

    TODO: implement those pre-activated ResNet variants besides 164 and 1001
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from chainercv.links import PickableSequentialChain
from chainercv.links import SEBlock

from chainerlp.links import Conv2DBNActiv, BNActivConv2D


class _BasicBlock(chainer.Chain):
    """ Basic block in ResNet-v2.

        BN and ReLU are inserted before CONV.
    """

    def __init__(self, in_channels, out_channels,
                 stride=1, pad=1, dilate=1, groups=1, initialW=None, bn_kwargs={},
                 residual_conv=False):
        """ CTOR. """
        super(_BasicBlock, self).__init__()

        with self.init_scope():
            self.conv1 = BNActivConv2D(in_channels, out_channels, ksize=3, stride=stride,
                                       pad=pad, dilate=dilate, nobias=True,
                                       initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv2 = BNActivConv2D(out_channels, out_channels, ksize=3, stride=1,
                                       pad=pad, dilate=dilate, nobias=True,
                                       initialW=initialW, bn_kwargs=bn_kwargs)

            if residual_conv:
                self.residual_conv = L.Convolution2D(in_channels, out_channels, ksize=1,
                                                     pad=0, stride=stride, nobias=True, initialW=initialW)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x

        h += residual
        # NOTE: no ReLU attached afterward

        return h


class _Bottleneck(chainer.Chain):
    """ Bottleneck in ResNet-v2 with pre-activation setting. """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, pad=1, dilate=1, groups=1,
                 initialW=None, bn_kwargs={}, residual_conv=False):
        """ CTOR """
        super(_Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilate = dilate
        self.groups = groups
        self.initialW = initialW
        self.bn_kwargs = bn_kwargs
        self.residual_conv = residual_conv

        with self.init_scope():
            self.conv1 = BNActivConv2D(in_channels, mid_channels, ksize=1,
                                       pad=0, nobias=True, initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv2 = BNActivConv2D(mid_channels, mid_channels, ksize=3,
                                       stride=stride, pad=pad, groups=groups, dilate=dilate, nobias=True,
                                       initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv3 = BNActivConv2D(mid_channels, out_channels, ksize=1,
                                       pad=0, nobias=True, initialW=initialW, bn_kwargs=bn_kwargs)

            if residual_conv:
                self.residual = L.Convolution2D(in_channels, out_channels, ksize=1,
                                                     pad=0, stride=stride, nobias=True, initialW=initialW)

    def forward(self, x):
        """ forward computation """
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        if hasattr(self, 'residual'):
            residual = self.residual(x)
        else:
            residual = x

        h += residual

        return h


class _ResBasicBlock(PickableSequentialChain):
    """ A layer in ResNet full of BasicBlocks. """

    def __init__(self, n_layer, in_channels, out_channels, stride=1, **kwargs):
        """ CTOR """
        super(_ResBasicBlock, self).__init__()

        with self.init_scope():
            self.a = _BasicBlock(in_channels, out_channels, stride=stride,
                                 residual_conv=True, **kwargs)  # NOTE: needs residual

            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                block = _BasicBlock(
                    out_channels, out_channels, stride=1, **kwargs)
                setattr(self, name, block)


class _ResBlock(PickableSequentialChain):
    """ ResNet-v2 block based on Bottleneck. """

    def __init__(self, n_layer, in_channels, mid_channels, out_channels, stride=1, **kwargs):
        """ CTOR. """
        super(_ResBlock, self).__init__()

        with self.init_scope():
            self.a = _Bottleneck(in_channels, mid_channels, out_channels,
                                 stride=stride, residual_conv=True, **kwargs)

            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                block = _Bottleneck(
                    out_channels, mid_channels, out_channels, stride=1, **kwargs)
                setattr(self, name, block)


class ResNetCIFARv2(PickableSequentialChain):
    """ Pre-activation based ResNet. """

    _blocks = {
        164: [18, 18, 18],
        1001: [111, 111, 111]}

    def __init__(self, n_layer, n_class=None, initialW=None, fc_kwargs={}):
        """ CTOR. """
        super(ResNetCIFARv2, self).__init__()

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)

        kwargs = {'initialW': initialW}
        blocks = self._blocks[n_layer]

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 16, ksize=3, pad=1, nobias=True, initialW=initialW)

            self.res2 = _ResBlock(blocks[0], 16, 16, 64, stride=1, **kwargs)
            self.res3 = _ResBlock(blocks[1], 64, 32, 128, stride=2, **kwargs)
            self.res4 = _ResBlock(blocks[2], 128, 64, 256, stride=2, **kwargs)

            # NOTE: these two subsequent layers are necessary to stablelize training
            self.bn = L.BatchNormalization(256)
            self.relu = lambda x: F.relu(x)

            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            # self.squeeze = lambda x: F.squeeze(x, axis=(2, 3))
            self.fc6 = L.Linear(256, n_class, **fc_kwargs)


class resnet164(ResNetCIFARv2):
    def __init__(self, n_class=None, **kwargs):
        super(resnet164, self).__init__(164, n_class=n_class, **kwargs)


class resnet1001(ResNetCIFARv2):
    def __init__(self, n_class=None, **kwargs):
        super(resnet1001, self).__init__(1001, n_class=n_class, **kwargs)
