""" Very simple convolutional neural networks """

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I

from chainercv.links import PickableSequentialChain

# from chainerlp.links.connection.ada_loss_convolution_2d import AdaLossConvolution2D
# from chainerlp.links.connection.ada_loss_linear import AdaLossLinear

# class AdaLossConvNet(PickableSequentialChain):
#     """ Use the lenet architecture """

#     def __init__(self,
#                  n_layers=4,
#                  n_class=10,
#                  init_scale=1.,
#                  ada_loss_cfg=None):
#         super().__init__()

#         if ada_loss_cfg is None:
#             ada_loss_cfg = {}

#         # setup the scale map
#         self.scale_map = np.ones(n_layers + 1, dtype=np.float32)
#         self.scale_map[-1] = init_scale
#         ada_loss_cfg['scale_map'] = self.scale_map

#         with self.init_scope():
#             self.reshape = lambda x: F.reshape(x, [-1, 1, 28, 28])

#             # Convolution layers
#             ada_loss_cfg['node_id'] = 0
#             self.conv1 = AdaLossConvolution2D(None,
#                                               16,
#                                               3,
#                                               stride=1,
#                                               pad=1,
#                                               initialW=I.HeNormal(),
#                                               ada_loss_cfg=ada_loss_cfg.copy())
#             self.pool1 = lambda x: F.max_pooling_2d(x, 2, stride=2)
#             self.relu1 = lambda x: F.relu(x)

#             ada_loss_cfg['node_id'] = 1
#             self.conv2 = AdaLossConvolution2D(None,
#                                               16,
#                                               3,
#                                               stride=1,
#                                               pad=1,
#                                               initialW=I.HeNormal(),
#                                               ada_loss_cfg=ada_loss_cfg.copy())
#             self.pool2 = lambda x: F.max_pooling_2d(x, 2, stride=2)
#             self.relu2 = lambda x: F.relu(x)

#             # fully connected layers
#             ada_loss_cfg['node_id'] = 2
#             self.fc3 = AdaLossLinear(None,
#                                      128,
#                                      ada_loss_cfg=ada_loss_cfg.copy(),
#                                      initialW=I.HeNormal())
#             self.relu3 = lambda x: F.relu(x)
#             ada_loss_cfg['node_id'] = 3
#             self.fc4 = AdaLossLinear(None,
#                                      n_class,
#                                      ada_loss_cfg=ada_loss_cfg.copy(),
#                                      initialW=I.HeNormal())
