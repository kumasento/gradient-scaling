import chainer

from ada_loss.chainer_impl.utils import scale_grad


class LossScaling(chainer.function_node.FunctionNode):
    """ Dummy loss scaling function node """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, inputs):
        return inputs

    def backward(self, indexes, grad_output):
        """ Wrapped backward computation """
        # scale all the gradients
        gs = [scale_grad(g, self.scale) for g in grad_output]
        # for g in gs:
        #     print(g.__dict__['loss_scale'])
        return gs


def loss_scaling(x, scale):
    """ The loss scaling function """
    is_list_input = False
    xs = (x,)

    if isinstance(x, list) or isinstance(x, tuple):
        xs = tuple(x)
        is_list_input = True

    ys = LossScaling(scale).apply(xs)

    if not is_list_input:
        return ys[0]
    return ys
