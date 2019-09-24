""" Convolution2D that supports AdaLoss """

import chainer
from chainer.utils import argument
from chainer.functions.connection import convolution_2d
import chainer.functions as F

from ..ada_loss import AdaLossChainer


class AdaLossConvolution2DFunction(convolution_2d.Convolution2DFunction):
    """ Convolution that supports adaptive loss scaling """

    def __init__(self,
                 stride=1,
                 pad=0,
                 cover_all=False,
                 ada_loss=None,
                 **kwargs):
        """ CTOR """
        super().__init__(stride=stride, pad=pad, cover_all=cover_all, **kwargs)

        dilate, _ = argument.parse_kwargs(kwargs, ('dilate', 1), ('groups', 1))

        self.ada_loss = ada_loss
        self.ada_loss.func = self
        self.ada_loss.func_params = {
            'stride': stride,
            'pad': pad,
            'cover_all': cover_all,
            'dilate': dilate,
        }

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        # NOTE: where to call the ada_loss function
        gy_, prev_scale = self.ada_loss.loss_scaling(gy, W)
        # gy_, prev_scale = gy, 1.0

        # xp = chainer.backend.get_array_module(x.array)
        # print(
        #     gy_.__dict__['loss_scale'],
        #     gy_.size,
        #     xp.count_nonzero(gy.array),
        #     xp.count_nonzero(gy_.array),
        #     gy.array.max(),
        #     gy_.array.max(),
        #     xp.abs(gy.array[gy.array > 0]).min(),
        #     xp.abs(gy_.array[gy_.array > 0]).min(),
        # )

        ret = []
        if 0 in indexes:
            xh, xw = x.shape[2:]
            gx = chainer.functions.deconvolution_2d(gy_,
                                                    W,
                                                    stride=(self.sy, self.sx),
                                                    pad=(self.ph, self.pw),
                                                    outsize=(xh, xw),
                                                    dilate=(self.dy, self.dx),
                                                    groups=self.groups)
            if (self.ada_loss.sanity_checker and
                self.ada_loss.recorder.current_iteration % self.ada_loss.sanity_checker.check_per_n_iter == 0):
                curr_iter = self.ada_loss.recorder.current_iteration
                gx_ = chainer.functions.deconvolution_2d(gy,
                                                         W,
                                                         stride=(self.sy, self.sx),
                                                         pad=(self.ph, self.pw),
                                                         outsize=(xh, xw),
                                                         dilate=(self.dy, self.dx),
                                                         groups=self.groups)
                gx2_ = chainer.functions.deconvolution_2d(F.cast(gy, 'float32'),
                                                         F.cast(W, 'float32'),
                                                         stride=(self.sy, self.sx),
                                                         pad=(self.ph, self.pw),
                                                         outsize=(xh, xw),
                                                         dilate=(self.dy, self.dx),
                                                         groups=self.groups)
                self.ada_loss.sanity_checker.check(gy, W, gx, gx_, gx2_, gy_.__dict__['loss_scale'], self.ada_loss.n_uf, curr_iter)
            # gx_ = chainer.functions.deconvolution_2d(gy,
            #                                          W,
            #                                          stride=(self.sy, self.sx),
            #                                          pad=(self.ph, self.pw),
            #                                          outsize=(xh, xw),
            #                                          dilate=(self.dy, self.dx),
            #                                          groups=self.groups)
            # print(gx.size, xp.count_nonzero(gx.array),
            #       xp.count_nonzero(gx_.array))
            # NOTE: here we pass the loss scale through gx's __dict__
            self.ada_loss.set_loss_scale(gx, self.ada_loss.grad_loss_scale(gy_))
            ret.append(gx)
        if 1 in indexes:
            gW, = convolution_2d.Convolution2DGradW(self).apply((x, gy))
            gW_ = self.ada_loss.get_unscaled_gradient(gW, prev_scale)
            ret.append(gW_)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=(0, 2, 3))
            gb_ = self.ada_loss.get_unscaled_gradient(gb, prev_scale)
            # ret.append(F.cast(gW_, W.dtype))
            ret.append(gb_)

        return ret


def ada_loss_convolution_2d(x,
                            W,
                            b=None,
                            stride=1,
                            pad=0,
                            cover_all=False,
                            ada_loss=None,
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
                                         ada_loss=ada_loss,
                                         dilate=dilate,
                                         groups=groups)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y