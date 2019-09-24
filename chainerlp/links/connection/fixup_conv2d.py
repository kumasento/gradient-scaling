""" Convolution2D function with Fixup support.
  https://github.com/hongyi-zhang/Fixup/blob/master/cifar/models/fixup_resnet_cifar.py
"""

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I


class FixupConv2D(chainer.Chain):
    """ Wraps Convolution2D by Fixup.

        Fixup works by adding a scalar bias to the input of convolution, optionally
        multiplying its output by another scalar multiplier, and then adding another
        bias scalar term. The final result might be passed through an activation func.
    """

    def __init__(self,
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
                 use_scale=True,
                 activ=F.relu):
        """ CTOR. """
        super(FixupConv2D, self).__init__()

        if initialW is None:  # NOTE: update it to zero initializer
            initialW = I.Zero()

        with self.init_scope():
            self.conv = L.Convolution2D(in_channels,
                                        out_channels,
                                        ksize=ksize,
                                        stride=stride,
                                        pad=pad,
                                        nobias=nobias,
                                        initialW=initialW,
                                        initial_bias=initial_bias,
                                        dilate=dilate,
                                        groups=groups)
            # bias term for conv input and output
            self.bias_in = chainer.Parameter(initializer=I.Zero(), shape=1)
            self.bias_out = chainer.Parameter(initializer=I.Zero(), shape=1)

            # NOTE: activ controls whether to use scale as well
            if use_scale or activ is None:
                self.scale = chainer.Parameter(initializer=I.One(), shape=1)
            else:
                self.scale = None

            # activation
            self.activ = activ

    def forward(self, x):
        """ forward """
        # core biased convolution function
        h = self.conv(x + self.bias_in)

        # optionally multiplies a scalar multiplier
        if self.scale is not None:
            h *= self.scale

        # bias out
        h += self.bias_out

        # optionally passes through an activation function
        if self.activ is None:
            return h
        else:
            return self.activ(h)
