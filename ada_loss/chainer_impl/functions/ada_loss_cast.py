""" Type casting with adaptive loss scaling support. """

import numpy as np

import chainer

LOG_GRAD=False

class AdaLossCastFunction(chainer.function_node.FunctionNode):
    def __init__(self, typ, ada_loss, lognormal=False):
        super().__init__()
        self.type = typ
        self.ada_loss = ada_loss
        self.ada_loss.func = self
        self.lognormal = lognormal

    def forward(self, x):
        self._in_type = x[0].dtype.type
        y =  x[0].astype(self.type, copy=False)
        return y,

    def backward(self, indexes, g):

        if np.dtype(self._in_type).kind != 'f':
            gx = None
        else:
            xp = chainer.backend.get_array_module(g[0].array)

            gx, prev_scale = self.ada_loss.loss_scaling(g[0], lognormal=self.lognormal)
            loss_scale = gx.__dict__['loss_scale']

            gx = ada_loss_cast(gx, self._in_type, self.ada_loss, lognormal=self.lognormal) 
            gx.__dict__['loss_scale'] = loss_scale

            if LOG_GRAD:
                nnz_1 = xp.count_nonzero(g[0].array) / g[0].array.size
                nnz_2 = xp.count_nonzero(gx.array) / gx.array.size

                if nnz_1 - nnz_2 > 0.2:
                    np.save('type_cast_grad.npy', xp.asnumpy(g[0].array))

                g_ = g[0].array
                nnz_g = g_[g_ != 0]

                if nnz_g.size != 0:
                    print('{} {} {:.6f} {:.6f} {:.6f} min={:e} max={:e}'.format(
                        loss_scale,
                        gx.array.size,
                        xp.asnumpy(nnz_1),
                        xp.asnumpy(nnz_2),
                        xp.asnumpy(nnz_1 - nnz_2),
                        xp.asnumpy(xp.abs(nnz_g).min()),
                        xp.asnumpy(xp.abs(nnz_g).max())))
        return gx,

def ada_loss_cast(x, typ, ada_loss, lognormal=False):
    if x.dtype == typ:
        if not chainer.config.enable_backprop:
            return chainer.as_variable(x)
    return AdaLossCastFunction(typ, ada_loss, lognormal=lognormal).apply((x,))[0]
