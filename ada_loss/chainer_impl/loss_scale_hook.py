""" Passing the loss scale and other status through the backward pass
    by FunctionHook. """

import chainer


class LossScaleHook(chainer.function_hook.FunctionHook):
    """  """

    def backward_preprocess(self, func, in_data, out_grad):
        """ """
        print('Preprocessing {} ...'.format(func.label))
        for i in range(len(in_data)):
            print('IN[{}]: {}'.format(
                i, in_data[i].shape if in_data[i] is not None else None))
            print(in_data[i])
        print(out_grad)

    def backward_postprocess(self, func, in_data, out_grad):
        """ """
        print(func.label)
        for i in range(len(in_data)):
            print('IN[{}]: {}'.format(
                i, in_data[i].shape if in_data[i] is not None else None))
            print(in_data[i])
        print(out_grad)