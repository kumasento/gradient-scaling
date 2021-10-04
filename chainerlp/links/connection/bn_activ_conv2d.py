""" Almost a replicate of Conv2DBNActiv in chainercv.links.connection. """

import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass


class BNActivConv2D(chainer.Chain):
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
        bn_kwargs={},
    ):
        super(BNActivConv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.dilate = dilate
        self.groups = groups
        self.nobias = nobias
        self.initialW = initialW
        self.initial_bias = initial_bias
        self.use_bn = True
        self.bn_kwargs = bn_kwargs
        self.activ = activ

        with self.init_scope():
            self.conv = Convolution2D(
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
            )

            # TODO: allow passing customized BN
            if "comm" in bn_kwargs:
                self.bn = MultiNodeBatchNormalization(in_channels, **bn_kwargs)
            else:
                self.bn = BatchNormalization(in_channels, **bn_kwargs)

    def __call__(self, x):
        h = self.bn(x)
        if self.activ is not None:
            h = self.activ(h)
        h = self.conv(h)

        return h
