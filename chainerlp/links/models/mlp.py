""" Definitions of different MLP functions """

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I

from chainercv.links import PickableSequentialChain

# from chainerlp.links import AdaLossLinear


class MLP(PickableSequentialChain):
    def __init__(self,
                 n_layer,
                 n_unit,
                 n_out,
                 n_in=None,
                 use_batchnorm=False,
                 **kwargs):
        super().__init__()

        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            for i in range(n_layer):
                # compute the input and output for each layer
                n_unit_ = n_unit if i != n_layer - 1 else n_out
                n_in_ = n_in if i == 0 else n_unit

                # node id is simply the current layer ID
                layer = L.Linear(n_in_,
                                 n_unit_,
                                 initialW=I.HeNormal(),
                                 **kwargs)
                name = 'l{}'.format(i + 1)
                setattr(self, name, layer)

                if use_batchnorm:
                    setattr(self, name + '_bn', L.BatchNormalization(n_unit_))

                # activation
                if i != n_layer - 1:
                    activ = lambda x: F.relu(x)
                    setattr(self, name + '_relu', activ)


# class AdaLossMLP(PickableSequentialChain):
#     def __init__(self,
#                  n_layer,
#                  n_unit,
#                  n_out,
#                  n_in=None,
#                  init_scale=1.0,
#                  use_batchnorm=False,
#                  **kwargs):
#         super().__init__()

#         # initialize the state to be shared by all sub-links
#         self.scale_map = np.ones(n_layer + 1, dtype=np.float32)
#         self.scale_map[-1] = init_scale

#         with self.init_scope():
#             # the size of the inputs to each layer will be inferred
#             for i in range(n_layer):
#                 # compute the input and output for each layer
#                 n_unit_ = n_unit if i != n_layer - 1 else n_out
#                 n_in_ = n_in if i == 0 else n_unit

#                 # node id is simply the current layer ID
#                 layer = AdaLossLinear(n_in_,
#                                       n_unit_,
#                                       node_id=i,
#                                       scale_map=self.scale_map,
#                                       initialW=I.HeNormal(),
#                                       **kwargs)
#                 name = 'l{}'.format(i + 1)
#                 setattr(self, name, layer)

#                 if use_batchnorm:
#                     setattr(self, name + '_bn', L.BatchNormalization(n_unit_))

#                 # activation
#                 if i != n_layer - 1:
#                     activ = lambda x: F.relu(x)
#                     setattr(self, name + '_relu', activ)