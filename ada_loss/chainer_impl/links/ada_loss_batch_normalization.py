import chainer
import chainer.links as L
from chainer import configuration
from chainer import functions
from chainer.utils import argument

from ada_loss.chainer_impl.functions.ada_loss_batch_normalization import ada_loss_batch_normalization, ada_loss_fixed_batch_normalization
from ada_loss.chainer_impl.ada_loss import AdaLossChainer


class AdaLossBatchNormalization(L.BatchNormalization):
    def __init__(self,
                 size=None,
                 decay=0.9,
                 eps=2e-5,
                 dtype=None,
                 use_gamma=True,
                 use_beta=True,
                 initial_gamma=None,
                 initial_beta=None,
                 axis=None,
                 initial_avg_mean=None,
                 initial_avg_var=None,
                 ada_loss_cfg=None):
        """ CTOR """
        super().__init__(size=size,
                         decay=decay,
                         eps=eps,
                         dtype=dtype,
                         use_gamma=use_gamma,
                         use_beta=use_beta,
                         initial_gamma=initial_gamma,
                         initial_beta=initial_beta,
                         axis=axis,
                         initial_avg_mean=initial_avg_mean,
                         initial_avg_var=initial_avg_var)

        self.ada_loss_cfg = {} if ada_loss_cfg is None else ada_loss_cfg

    def forward(self, x, **kwargs):
        """forward(self, x, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluation during training, and normalizes the
        input using batch statistics.
        Args:
            x (~chainer.Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        finetune, = argument.parse_kwargs(
            kwargs, ('finetune', False),
            test='test argument is not supported anymore. '
            'Use chainer.using_config')

        if self.avg_mean is None:
            param_shape = tuple(
                [d for i, d in enumerate(x.shape) if i not in self.axis])
            self._initialize_params(param_shape)

        gamma = self.gamma
        if gamma is None:
            with chainer.using_device(self.device):
                gamma = self.xp.ones(self.avg_mean.shape,
                                     dtype=self._highprec_dtype)

        beta = self.beta
        if beta is None:
            with chainer.using_device(self.device):
                beta = self.xp.zeros(self.avg_mean.shape,
                                     dtype=self._highprec_dtype)

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            avg_mean = self.avg_mean
            avg_var = self.avg_var

            if chainer.config.in_recomputing:
                # Do not update statistics when extra forward computation is
                # called.
                if finetune:
                    self.N -= 1  # Revert the count
                avg_mean = None
                avg_var = None

            ret = ada_loss_batch_normalization(x,
                                               gamma,
                                               beta,
                                               eps=self.eps,
                                               running_mean=avg_mean,
                                               running_var=avg_var,
                                               decay=decay,
                                               axis=self.axis,
                                               ada_loss_cfg=self.ada_loss_cfg)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = self.avg_mean
            var = self.avg_var
            ret = ada_loss_fixed_batch_normalization(
                x,
                gamma,
                beta,
                mean,
                var,
                self.eps,
                axis=self.axis,
                ada_loss_cfg=self.ada_loss_cfg)
        return ret
