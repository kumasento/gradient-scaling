""" The identity transition in Fixup resnet. """

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

from chainer.backend import cuda


class FixupIdentity(chainer.Chain):
    """ Average pooling followed by zero concat.
        Used as an identity mapping in Fixup based models. 
    """

    def __init__(self, stride):
        super(FixupIdentity, self).__init__()

        self.h_zero = None
        self.stride = stride

    def forward(self, x):
        """ """
        h = F.average_pooling_2d(x, ksize=1, stride=self.stride)
        if self.stride == 1:
            return h

        return F.concat([h, self.get_zeros_like(h)], axis=1)
        # pad_width = ((0, 0), (0, h.shape[1]), (0, 0), (0, 0))
        # return F.pad(h, pad_width, mode='constant', constant_values=0)

    def get_zeros_like(self, h):
        """ Return a new array or a cached one """
        if self.h_zero is None or self.h_zero.shape != h.shape:
            # print('Creating new array for {}'.format(h.shape))
            xp = chainer.backend.get_array_module(h)

            # create the array
            self.h_zero = chainer.Variable(xp.zeros_like(h.data))
            self.h_zero.to_device(chainer.backend.get_device_from_array(
                h.data))

        return self.h_zero
