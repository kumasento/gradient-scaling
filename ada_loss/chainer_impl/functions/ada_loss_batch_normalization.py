""" Wrapped batch norm """

import chainer
import chainer.functions as F
from chainer.utils import argument
from chainer.functions.normalization import batch_normalization

from ada_loss.chainer_impl.ada_loss import AdaLossChainer


class AdaLossBatchNormalization(batch_normalization.BatchNormalization):
    """ """

    def __init__(self,
                 eps=2e-5,
                 mean=None,
                 var=None,
                 decay=0.9,
                 axis=None,
                 ada_loss_cfg=None):
        super().__init__(eps=eps, mean=mean, var=var, decay=decay, axis=axis)

        if ada_loss_cfg is None:
            ada_loss_cfg = {}
        self.ada_loss = AdaLossChainer(**ada_loss_cfg)

    def backward(self, indexes, grad_outputs):
        x, gamma = self.get_retained_inputs()
        gy, = grad_outputs

        if self.use_ideep:
            assert self.var is not None
            var = self.var
        else:
            var = None

        f = batch_normalization.BatchNormalizationGrad(
            self.eps, self.use_cudnn, self.mode, self.expander, self.axis,
            self.mean, var, self.inv_std, self.key_axis)
        gx, ggamma, gbeta = f.apply((x, gamma, gy))

        # update the loss scale
        prev_scale = self.ada_loss.get_prev_scale(gy)
        # NOTE: the numerical stability here?
        # NOTE: efficiency?
        ggamma_ = self.ada_loss.get_unscaled_gradient(ggamma,
                                                      prev_scale,
                                                      dtype=ggamma.dtype)
        gbeta_ = self.ada_loss.get_unscaled_gradient(gbeta,
                                                     prev_scale,
                                                     dtype=ggamma.dtype)
        self.ada_loss.set_loss_scale(gx, prev_scale)  # pass along

        return gx, ggamma_, gbeta_


def ada_loss_batch_normalization(x, gamma, beta, ada_loss_cfg=None, **kwargs):
    eps, running_mean, running_var, decay, axis = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None), ('running_var', None),
        ('decay', 0.9), ('axis', None),
        train='train argument is not supported anymore. '
        'Use chainer.using_config')

    return AdaLossBatchNormalization(eps,
                                     running_mean,
                                     running_var,
                                     decay,
                                     axis,
                                     ada_loss_cfg=ada_loss_cfg).apply(
                                         (x, gamma, beta))[0]


class AdaLossFixedBatchNormalization(
        batch_normalization.FixedBatchNormalization):
    """ Wrapping the fixed batch normalization function """

    def __init__(self, eps=2e-5, axis=None, ada_loss_cfg=None):
        super().__init__(eps=eps, axis=axis)

        if ada_loss_cfg is None:
            ada_loss_cfg = {}
        self.ada_loss = AdaLossChainer(**ada_loss_cfg)

    def backward(self, indexes, grad_outputs):
        """ wrap around the original backward function """
        x, gamma, mean, var = self.get_retained_inputs()
        gy, = grad_outputs
        f = batch_normalization.FixedBatchNormalizationGrad(
            self.eps, self.expander, self.axis, self.inv_std, self.inv_var)
        gx, ggamma, gbeta, gmean, gvar = f.apply((x, gamma, mean, var, gy))

        prev_scale = self.ada_loss.get_prev_scale(gy)
        ggamma_ = self.ada_loss.get_unscaled_gradient(ggamma,
                                                      prev_scale,
                                                      dtype=ggamma.dtype)
        gbeta_ = self.ada_loss.get_unscaled_gradient(gbeta,
                                                     prev_scale,
                                                     dtype=gbeta.dtype)
        gmean_ = self.ada_loss.get_unscaled_gradient(gmean,
                                                     prev_scale,
                                                     dtype=gmean.dtype)
        gvar_ = self.ada_loss.get_unscaled_gradient(gvar,
                                                    prev_scale,
                                                    dtype=gvar.dtype)
        self.ada_loss.set_loss_scale(gx, prev_scale)  # pass along

        return gx, ggamma_, gbeta_, gmean_, gvar_


def ada_loss_fixed_batch_normalization(x,
                                       gamma,
                                       beta,
                                       mean,
                                       var,
                                       eps=2e-5,
                                       axis=None,
                                       ada_loss_cfg=None):
    """ """
    return AdaLossFixedBatchNormalization(eps, axis,
                                          ada_loss_cfg=ada_loss_cfg).apply(
                                              (x, gamma, beta, mean, var))[0]
