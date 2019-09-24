""" Various VGG based models. """

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass

from chainercv.links import PickableSequentialChain

from chainerlp.links import Conv2DBNActiv
# from chainerlp.links import AdaLossLinear, AdaLossConvolution2D


class Block(chainer.Chain):
    """A convolution, batch norm, ReLU block.

    Batch norm can be turned off.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 pad=1,
                 use_batchnorm=False,
                 initialW=None,
                 bn_kwargs={},
                 conv2d=None,
                 **kwargs):
        super(Block, self).__init__()
        with self.init_scope():
            if conv2d is None:
                conv2d = L.Convolution2D

            self.conv = conv2d(in_channels,
                               out_channels,
                               ksize,
                               pad=pad,
                               initialW=initialW,
                               nobias=False,
                               **kwargs)
            if use_batchnorm:
                # TODO: allow passing customized BN
                if 'comm' in bn_kwargs:
                    self.bn = MultiNodeBatchNormalization(
                        out_channels, **bn_kwargs)
                else:
                    self.bn = L.BatchNormalization(out_channels, **bn_kwargs)
            self.relu = lambda x: F.relu(x)

    def forward(self, x):
        h = self.conv(x)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        return self.relu(h)


class VGGBlock(PickableSequentialChain):
    def __init__(self,
                 n_layer,
                 in_channels,
                 out_channels,
                 use_batchnorm=False,
                 initialW=None,
                 bn_kwargs={},
                 **kwargs):
        super(VGGBlock, self).__init__()

        _KSIZE = 3
        with self.init_scope():
            self.a = Block(in_channels,
                           out_channels,
                           _KSIZE,
                           use_batchnorm=use_batchnorm,
                           initialW=initialW,
                           bn_kwargs=bn_kwargs,
                           **kwargs)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                block = Block(out_channels,
                              out_channels,
                              _KSIZE,
                              use_batchnorm=use_batchnorm,
                              initialW=initialW,
                              bn_kwargs=bn_kwargs,
                              **kwargs)
                setattr(self, name, block)

            self.pool = lambda x: F.max_pooling_2d(x, (2, 2), stride=2)


class VGGNetCIFAR(PickableSequentialChain):
    """ A VGG-style network for very small images.

    Same architecture as in
        https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/vgg.py 
    """

    _blocks = {
        11: [1, 1, 2, 2, 2],
        13: [2, 2, 2, 2, 2],
        16: [2, 2, 3, 3, 3],
        19: [2, 2, 4, 4, 4],
    }

    def __init__(self,
                 n_layer,
                 n_class=None,
                 use_batchnorm=False,
                 initialW=None,
                 fc_kwargs={},
                 **kwargs):
        super(VGGNetCIFAR, self).__init__()

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initialW  # initializers.Normal(scale=0.01)
        kwargs.update({'initialW': initialW, 'use_batchnorm': use_batchnorm})

        blocks = self._blocks[n_layer]
        with self.init_scope():
            self.block1 = VGGBlock(blocks[0], 3, 64, **kwargs)
            self.block2 = VGGBlock(blocks[1], 64, 128, **kwargs)
            self.block3 = VGGBlock(blocks[2], 128, 256, **kwargs)
            self.block4 = VGGBlock(blocks[3], 256, 512, **kwargs)
            self.block5 = VGGBlock(blocks[4], 512, 512, **kwargs)
            self.squeeze = lambda x: F.squeeze(x, axis=(2, 3))
            self.fc = L.Linear(512, n_class, nobias=False, **fc_kwargs)


# class VGGBlockAdaLoss(PickableSequentialChain):
#     def __init__(self,
#                  n_layer,
#                  in_channels,
#                  out_channels,
#                  use_batchnorm=False,
#                  initialW=None,
#                  bn_kwargs={},
#                  start_id=0,
#                  **kwargs):
#         super().__init__()
#         assert 'ada_loss_cfg' in kwargs
#         kwargs['conv2d'] = AdaLossConvolution2D
#
#         _KSIZE = 3
#         with self.init_scope():
#             kwargs['ada_loss_cfg']['node_id'] = start_id
#             self.a = Block(in_channels,
#                            out_channels,
#                            _KSIZE,
#                            use_batchnorm=use_batchnorm,
#                            initialW=initialW,
#                            bn_kwargs=bn_kwargs,
#                            **kwargs)
#
#             for i in range(n_layer - 1):
#                 kwargs['ada_loss_cfg']['node_id'] = start_id + i + 1
#                 name = 'b{}'.format(i + 1)
#                 block = Block(out_channels,
#                               out_channels,
#                               _KSIZE,
#                               use_batchnorm=use_batchnorm,
#                               initialW=initialW,
#                               bn_kwargs=bn_kwargs,
#                               **kwargs)
#                 setattr(self, name, block)
#
#             self.pool = lambda x: F.max_pooling_2d(x, (2, 2), stride=2)

# class VGGNetCIFARAdaLoss(PickableSequentialChain):
#     """ Added AdaLoss """

#     _blocks = {
#         11: [1, 1, 2, 2, 2],
#         13: [2, 2, 2, 2, 2],
#         16: [2, 2, 3, 3, 3],
#         19: [2, 2, 4, 4, 4],
#     }

#     def __init__(self,
#                  n_layer,
#                  n_class=None,
#                  use_batchnorm=False,
#                  initialW=None,
#                  fc_kwargs={},
#                  init_scale=16.,
#                  **kwargs):
#         super().__init__()

#         self.scale_map = np.ones(n_layer + 1, dtype=np.float32)
#         self.scale_map[-1] = init_scale

#         if initialW is None:
#             initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
#         if 'initialW' not in fc_kwargs:
#             fc_kwargs['initialW'] = initialW  # initializers.Normal(scale=0.01)

#         kwargs.update({
#             'initialW': initialW,
#             'use_batchnorm': use_batchnorm,
#         })
#         kwargs['ada_loss_cfg']['scale_map'] = self.scale_map

#         blocks = self._blocks[n_layer]
#         with self.init_scope():
#             start_id = 0
#             self.block1 = VGGBlockAdaLoss(blocks[0],
#                                           3,
#                                           64,
#                                           start_id=start_id,
#                                           **kwargs)
#             start_id += blocks[0]
#             self.block2 = VGGBlockAdaLoss(blocks[1],
#                                           64,
#                                           128,
#                                           start_id=start_id,
#                                           **kwargs)
#             start_id += blocks[1]
#             self.block3 = VGGBlockAdaLoss(blocks[2],
#                                           128,
#                                           256,
#                                           start_id=start_id,
#                                           **kwargs)
#             start_id += blocks[2]
#             self.block4 = VGGBlockAdaLoss(blocks[3],
#                                           256,
#                                           512,
#                                           start_id=start_id,
#                                           **kwargs)
#             start_id += blocks[3]
#             self.block5 = VGGBlockAdaLoss(blocks[4],
#                                           512,
#                                           512,
#                                           start_id=start_id,
#                                           **kwargs)
#             start_id += blocks[4]
#             self.fc = AdaLossLinear(512,
#                                     n_class,
#                                     nobias=False,
#                                     node_id=start_id,
#                                     **fc_kwargs)


class vgg11(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg11, self).__init__(11, n_class=n_class, **kwargs)


# class vgg11_ada_loss(VGGNetCIFARAdaLoss):
#     def __init__(self, n_class=None, **kwargs):
#         super().__init__(11, n_class=n_class, **kwargs)


class vgg13(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg13, self).__init__(13, n_class=n_class, **kwargs)


class vgg16(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg16, self).__init__(16, n_class=n_class, **kwargs)


class vgg19(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg19, self).__init__(19, n_class=n_class, **kwargs)


class vgg11_bn(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg11_bn, self).__init__(11,
                                       n_class=n_class,
                                       use_batchnorm=True,
                                       **kwargs)


class vgg13_bn(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg13_bn, self).__init__(13,
                                       n_class=n_class,
                                       use_batchnorm=True,
                                       **kwargs)


class vgg16_bn(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg16_bn, self).__init__(16,
                                       n_class=n_class,
                                       use_batchnorm=True,
                                       **kwargs)


class vgg19_bn(VGGNetCIFAR):
    def __init__(self, n_class=None, **kwargs):
        super(vgg19_bn, self).__init__(19,
                                       n_class=n_class,
                                       use_batchnorm=True,
                                       **kwargs)
