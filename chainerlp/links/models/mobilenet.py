""" Implementation of MobileNet-v1 in Chainer.

Ref: https://github.com/peisuke/DeepLearningSpeedComparison/blob/master/chainer/mobilenet/predict.py
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from chainercv.links import PickableSequentialChain

from chainerlp.links import Conv2DBNActiv


class DepthwiseSeparableConv2D(PickableSequentialChain):
    """ Basic building block for MobileNet-v1 """

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, initialW=None):
        """ CTOR. """
        super(DepthwiseSeparableConv2D, self).__init__()

        with self.init_scope():
            self.conv1 = L.DepthwiseConvolution2D(
                in_channels, 1, ksize, stride=stride, pad=1, nobias=True, initialW=initialW)
            self.bn1 = L.BatchNormalization(in_channels)
            self.relu1 = lambda x: F.relu(x)
            self.conv2 = L.Convolution2D(
                in_channels, out_channels, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW)
            self.bn2 = L.BatchNormalization(out_channels)
            self.relu2 = lambda x: F.relu(x)


class MobileNetV1Block(PickableSequentialChain):

    def __init__(self, n_layer, in_channels, out_channels, ksize=3, stride=1, initialW=None):
        """ CTOR. """
        super(MobileNetV1Block, self).__init__()
        with self.init_scope():
            self.a = DepthwiseSeparableConv2D(
                in_channels, out_channels, ksize=ksize, stride=stride, initialW=initialW)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i)
                block = DepthwiseSeparableConv2D(
                    out_channels, out_channels, ksize=ksize, stride=1, initialW=initialW)
                setattr(self, name, block)


class MobileNetV1(PickableSequentialChain):
    """ V1 of MobileNet for CIFAR.
    TODO: build one for ImageNet
    """

    def __init__(self, n_class=None, initialW=None):
        super(MobileNetV1, self).__init__()

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {'initialW': initialW}

        with self.init_scope():
            # first layer
            self.conv1 = L.Convolution2D(
                3, 32, ksize=3, stride=1, pad=1, nobias=True, **kwargs)
            self.bn1 = L.BatchNormalization(32)

            # the rest of the laters
            self.block1 = MobileNetV1Block(1, 32, 64, stride=1, **kwargs)
            self.block2 = MobileNetV1Block(2, 64, 128, stride=2, **kwargs)
            self.block3 = MobileNetV1Block(2, 128, 256, stride=2, **kwargs)
            self.block4 = MobileNetV1Block(6, 256, 512, stride=2, **kwargs)
            self.block5 = MobileNetV1Block(2, 512, 1024, stride=2, **kwargs)

            # avg pooling
            self.pool = lambda x: F.average_pooling_2d(x, ksize=2)
            self.fc = L.Linear(1024, n_class)


class mobilenet_v1(MobileNetV1):
    def __init__(self, n_class=None, **kwargs):
        super(mobilenet_v1, self).__init__(n_class=n_class, **kwargs)
