import numpy as np
import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from ada_loss.chainer_impl.functions.ada_loss_branch import AdaLossBranch
from ada_loss.chainer_impl.ada_loss import AdaLossChainer
from ada_loss.chainer_impl.functions.ada_loss_cast import ada_loss_cast
from ada_loss.chainer_impl.links.ada_loss_convolution_2d import AdaLossConvolution2D


def post_loc(x):
    """ Post loc convolution """
    y = F.transpose(x, (0, 2, 3, 1))
    y = F.reshape(y, (y.shape[0], -1, 4))
    # y = F.cast(y, 'float32')
    return y


def post_conf(mb_conf, n_class):
    """ """
    mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
    mb_conf = F.reshape(mb_conf, (mb_conf.shape[0], -1, n_class))
    # mb_conf = F.cast(mb_conf, 'float32')
    return mb_conf


class Multibox(chainer.Chain):
    """Multibox head of Single Shot Multibox Detector.

    This is a head part of Single Shot Multibox Detector [#]_.
    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(self, n_class, aspect_ratios, initialW=None, initial_bias=None):
        super(Multibox, self).__init__()

        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {"initialW": initialW, "initial_bias": initial_bias}

        with self.init_scope():
            # with chainer.using_config('dtype', 'float32'):
            for i, ar in enumerate(aspect_ratios):
                n = (len(ar) + 1) * 2
                loc_name = "loc_{}".format(i)
                conf_name = "conf_{}".format(i)

                setattr(self, loc_name, L.Convolution2D(n * 4, 3, pad=1, **init))
                setattr(
                    self, conf_name, L.Convolution2D(n * self.n_class, 3, pad=1, **init)
                )

            self.concat_locs = lambda xs: F.concat(xs, axis=1)
            self.concat_confs = lambda xs: F.concat(xs, axis=1)
            self.post_loc = lambda x: post_loc(x)
            self.post_conf = lambda x: post_conf(x, self.n_class)

        self.tc_locs = [None] * len(aspect_ratios)
        self.tc_confs = [None] * len(aspect_ratios)

    def forward(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = []
        mb_confs = []

        dtype = chainer.global_config.dtype

        for i, x in enumerate(xs):
            # TODO: can we don't refer to AdaLossBranch here? Maybe turn it to a
            # general forward function?
            x1, x2 = AdaLossBranch().apply((x,))
            loc = getattr(self, "loc_{}".format(i))
            mb_loc = loc(x1)
            mb_loc = self.post_loc(mb_loc)

            conf = getattr(self, "conf_{}".format(i))
            mb_conf = conf(x2)
            mb_conf = self.post_conf(mb_conf)

            if dtype != np.float32:
                if not isinstance(loc, AdaLossConvolution2D):
                    mb_loc = F.cast(mb_loc, "float32")
                    mb_conf = F.cast(mb_conf, "float32")
                else:
                    if self.tc_locs[i] is None:
                        self.tc_locs[i] = AdaLossChainer(**loc.ada_loss_cfg)
                    if self.tc_confs[i] is None:
                        self.tc_confs[i] = AdaLossChainer(**loc.ada_loss_cfg)
                    mb_loc = ada_loss_cast(mb_loc, "float32", self.tc_locs[i])
                    mb_conf = ada_loss_cast(
                        mb_conf, "float32", self.tc_confs[i], lognormal=True
                    )

            mb_locs.append(mb_loc)
            mb_confs.append(mb_conf)

        mb_locs = self.concat_locs(mb_locs)
        mb_confs = self.concat_confs(mb_confs)

        return mb_locs, mb_confs
