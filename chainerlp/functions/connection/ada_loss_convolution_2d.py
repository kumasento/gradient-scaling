""" Convolution2D that supports AdaLoss """

import chainer
from chainer.utils import argument
from chainer.functions.connection import convolution_2d

from chainerlp.ada_loss.ada_loss_chainer import AdaLossChainer


class AdaLossConvolution2DFunction(convolution_2d.Convolution2DFunction):
    """ Convolution that supports adaptive loss scaling """

    def __init__(self,
                 stride=1,
                 pad=0,
                 cover_all=False,
                 ada_loss_cfg=None,
                 **kwargs):
        """ CTOR """
        super().__init__(stride=stride, pad=pad, cover_all=cover_all, **kwargs)

        # NOTE: initialize the loss scaler
        if ada_loss_cfg is None:
            ada_loss_cfg = {}

        dilate, _ = argument.parse_kwargs(kwargs, ('dilate', 1), ('groups', 1))
        ada_loss_cfg['func_params'] = {
            'stride': stride,
            'pad': pad,
            'cover_all': cover_all,
            'dilate': dilate,
        }
        self.ada_loss = AdaLossChainer(**ada_loss_cfg)

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        # NOTE: where to call the ada_loss function
        s_gy, u_gy = self.ada_loss.loss_scaling(gy, W)

        ret = []
        if 0 in indexes:
            xh, xw = x.shape[2:]
            gx = chainer.functions.deconvolution_2d(s_gy,
                                                    W,
                                                    stride=(self.sy, self.sx),
                                                    pad=(self.ph, self.pw),
                                                    outsize=(xh, xw),
                                                    dilate=(self.dy, self.dx),
                                                    groups=self.groups)
            ret.append(gx)
        if 1 in indexes:
            gW, = convolution_2d.Convolution2DGradW(self).apply((x, u_gy))
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(u_gy, axis=(0, 2, 3))
            ret.append(gb)

        return ret


def ada_loss_convolution_2d(x,
                            W,
                            b=None,
                            stride=1,
                            pad=0,
                            cover_all=False,
                            ada_loss_cfg=None,
                            **kwargs):
    """ The convolution_2d function that supports loss scaling """
    dilate, groups = argument.parse_kwargs(
        kwargs, ('dilate', 1), ('groups', 1),
        deterministic='deterministic argument is not supported anymore. '
        'Use chainer.using_config(\'cudnn_deterministic\', value) '
        'context where value is either `True` or `False`.')

    fnode = AdaLossConvolution2DFunction(stride,
                                         pad,
                                         cover_all,
                                         ada_loss_cfg=ada_loss_cfg,
                                         dilate=dilate,
                                         groups=groups)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y