""" Implementation of Wide-ResNet in Chainer.

Ref:
https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
https://github.com/mitmul/chainer-cifar10/blob/master/models/wide_resnet.py
https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py
"""


import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from chainercv.links import PickableSequentialChain


class _BasicBlock(chainer.Chain):
    """ Basic block in WRN. """

    def __init__(self, in_channels, out_channels, stride, drop_rate=0.0, initialW=None):
        """ CTOR. """
        self.drop_rate = drop_rate

        super(_BasicBlock, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, ksize=3, stride=stride, pad=1, nobias=True, initialW=initialW)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW)
            self.bn1 = L.BatchNormalization(in_channels)
            self.bn2 = L.BatchNormalization(out_channels)

            if in_channels != out_channels:
                # Create a shortcut mapping with Convolution when the input and output
                # channels don't match
                self.residual_conv = L.Convolution2D(
                    in_channels, out_channels, ksize=1, stride=stride, pad=0, nobias=True, initialW=initialW)

    def forward(self, x):
        """ Forward computation """
        o1 = F.relu(self.bn1(x))
        y = self.conv1(o1)
        o2 = F.relu(self.bn2(y))
        if self.drop_rate > 0:
            o2 = F.dropout(o2, ratio=self.drop_rate)
        z = self.conv2(o2)

        # The connection points for residual_conv input and the
        # identity mapping are different.
        if hasattr(self, 'residual_conv'):
            return z + self.residual_conv(o1)
        else:
            return z + x


class _WideResBlock(PickableSequentialChain):
    """ A block in WRN that consists of multiple _BasicBlock """

    def __init__(self, n_layer, in_channels, out_channels, stride, drop_rate=0.0, initialW=None):
        """ CTOR. """
        super(_WideResBlock, self).__init__()

        with self.init_scope():
            self.a = _BasicBlock(in_channels, out_channels,
                                 stride, drop_rate=drop_rate, initialW=initialW)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                block = _BasicBlock(out_channels, out_channels,
                                    1, drop_rate=drop_rate, initialW=initialW)
                setattr(self, name, block)


class WideResNet(PickableSequentialChain):
    """ A full WRN module. """

    def __init__(self, depth, n_class, widen_factor=1, drop_rate=0.0, initialW=None):
        """ CTOR. """
        super(WideResNet, self).__init__()

        k = widen_factor

        assert (depth - 4) % 6 == 0, 'Depth should be 6n + 4'
        n = (depth - 4) // 6
        n_channel = [16, 16 * k, 32 * k, 64 * k]

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {'initialW': initialW}

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, n_channel[0], ksize=3, stride=1, pad=1, nobias=True, **kwargs)

            self.wide2 = _WideResBlock(
                n, n_channel[0], n_channel[1], 1, drop_rate=drop_rate, **kwargs)
            self.wide3 = _WideResBlock(
                n, n_channel[1], n_channel[2], 2, drop_rate=drop_rate, **kwargs)
            self.wide4 = _WideResBlock(
                n, n_channel[2], n_channel[3], 2, drop_rate=drop_rate, **kwargs)

            self.bn = L.BatchNormalization(n_channel[3])
            self.relu = lambda x: F.relu(x)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(n_channel[3], n_class, **kwargs)


class wrn_28_10(WideResNet):
    """ Wide-ResNet with 28 layers and 10 as widening factor  """

    def __init__(self, n_class=None, drop_rate=0.3):
        """ CTOR. """
        super(wrn_28_10, self).__init__(
            28, n_class, widen_factor=10, drop_rate=drop_rate)

class wrn_28_20(WideResNet):
    """ Wide-ResNet with 28 layers and 20 as widening factor  """

    def __init__(self, n_class=None, drop_rate=0.3):
        """ CTOR. """
        super(wrn_28_20, self).__init__(
            28, n_class, widen_factor=20, drop_rate=drop_rate)

class wrn_40_4(WideResNet):
    """ Wide-ResNet with 40 layers and 4 as widening factor  """

    def __init__(self, n_class=None, drop_rate=0.3):
        """ CTOR. """
        super(wrn_40_4, self).__init__(
            40, n_class, widen_factor=4, drop_rate=drop_rate)


class wrn_16_8(WideResNet):
    """ Wide-ResNet with 16 layers and 8 as widening factor  """

    def __init__(self, n_class=None, drop_rate=0.3):
        """ CTOR. """
        super(wrn_16_8, self).__init__(
            16, n_class, widen_factor=8, drop_rate=drop_rate)
