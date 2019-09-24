""" Monitor the progress of adaptive loss scaling training """

import os
import pandas as pd
import chainer
import pickle


class AdaLossMonitor(chainer.function_hook.FunctionHook):
    """ Definition of the hook that records the statistics of activation. """

    name = 'AdaLossMonitor'

    def __init__(self,
                 trainer=None,
                 sample_iterations=None,
                 sample_per_n_iter=None,
                 snapshot_per_n_iter=None,
                 snapshot_dir=None,
                 includes=None,
                 verbose=False):
        """ """
        self.trainer = trainer
        self.history = []

        if sample_iterations is None:
            sample_iterations = []
        self.sample_iterations = sample_iterations
        self.sample_per_n_iter = sample_per_n_iter

        self.snapshot_per_n_iter = snapshot_per_n_iter
        self.snapshot_dir = snapshot_dir
        if snapshot_dir is not None:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        self.counter = 0

        self.includes = includes
        self.verbose = verbose

    @property
    def optim(self):
        optims = self.trainer.updater.get_all_optimizers()
        return optims['main']

    @property
    def loss_scale(self):
        return self.optim._loss_scale

    @property
    def current_iteration(self):
        return self.trainer.updater.iteration

    def is_recording(self):
        """ Whether we are recording samples """
        if self.sample_per_n_iter is not None:
            return self.current_iteration % self.sample_per_n_iter == 0
        else:
            return self.current_iteration in self.sample_iterations

    def is_included(self, func):
        """ Check whether the given function should be recorded """
        if self.includes is None:
            return True
        label = func.label
        for inc in self.includes:
            if inc in label:
                return True
        return False

    def forward_preprocess(self, function, in_data):
        """ Monitor the gradient """
        if self.is_included(function):
            return self.process(function, in_data)

        # if 'Grad' in function.label:
        #     return self.process_gradient(function, in_data)
        # if 'Linear' in function.label:
        #     return self.process_linear(function, in_data)
        # if 'Softmax' in function.label:
        #     return self.process_softmax(function, in_data)
        # if 'Convolution' in function.label:
        #     return self.process_convolution_2d(function, in_data)
        # if 'Deconvolution' in function.label:
        #     return self.process_deconvolution_2d(function, in_data)

        # otherwise, nothing to do
        return

    def process(self, func, data):
        """ process the forward computation """
        # record various statistics
        if self.is_recording():
            if self.verbose:
                print('==> Processing function {} {} ...'.format(
                    func.label, data[0].shape))
            for i, d in enumerate(data):
                if d is not None and d.size > 0:
                    sample = self.get_sample(d)
                    self.history.append([
                        self.current_iteration,
                        func.label,
                        i,
                        *sample,
                    ])
        if (self.snapshot_per_n_iter is not None and 
            self.current_iteration % self.snapshot_per_n_iter == 0):
            for i, d in enumerate(data):
                if d is not None and d.size > 0:
                    fp = os.path.join(
                        self.snapshot_dir,
                        'iter_{}_cnt_{}_func_{}_idx_{}.pkl'.format(
                            self.current_iteration, self.counter, func.label, i))
                    xp = chainer.backend.get_array_module(d)
                    with open(fp, 'wb') as f:
                        pickle.dump(xp.asnumpy(d), f)
            self.counter += 1



    def get_sample(self, data):
        """ get sampling result from data """
        xp = chainer.backend.get_array_module(data)
        data_ = data.astype('float32')

        sample = [
            # size
            data.size,
            # nnz
            xp.count_nonzero(data).item(),
            # min, max
            xp.abs(data).min().item(),
            xp.abs(data).max().item(),
            # mean
            data.mean().item(),
            # std
            data_.std().item(),
            # norm
            xp.linalg.norm(data_).item()
        ]

        return sample

    def process_linear(self, func, data):
        """ Simply the forward update """
        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing forward for {} ...'.format(func.label))

        for i in range(len(data)):
            print('{:20s}\t{}\t{}\t{}\tnorm={}'.format(
                str(data[i].shape), xp.count_nonzero(data[i]), data[i].min(),
                data[i].max(), xp.linalg.norm(data[i].astype('float32'))))
        # print(xp.dot(data[0], data[1].T))

    def process_softmax(self, func, data):
        """ Simply the forward update """
        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing forward for {} ...'.format(func.label))

        for i in range(len(data)):
            print('{:20s}\t{}\t{}\t{}\t{}'.format(str(data[i].shape),
                                                  xp.count_nonzero(data[i]),
                                                  data[i].min(), data[i].max(),
                                                  xp.linalg.norm(data[i])))

    def process_convolution_2d(self, func, data):
        """ Simply the forward update """
        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing forward for {} ...'.format(func.label))

        for i in range(len(data)):
            print('{:20s}\t{}\t{}\t{}\tnorm={}'.format(
                str(data[i].shape), xp.count_nonzero(data[i]), data[i].min(),
                data[i].max(), xp.linalg.norm(data[i])))
        # print(xp.dot(data[0], data[1].T))

    def process_gradient(self, func, data):
        """ Process the input data to a gradient function. """
        # gradient functions are implemented as a forward function in Chainer
        assert 'Grad' in func.label

        if 'Linear' in func.label:
            return self.process_linear_grad(func, data)
        if 'Softmax' in func.label:
            return self.process_softmax_grad(func, data)
        if 'Convolution' in func.label:
            return self.process_convolution_2d_weight_grad(func, data)

    def process_linear_grad(self, func, data):
        """ Process the linear function's gradient update """
        assert 'Linear' in func.label

        if 'Weight' in func.label:
            return self.process_linear_weight_grad(func, data)
        if 'Data' in func.label:
            return self.process_linear_data_grad(func, data)

    def process_softmax_grad(self, func, data):
        assert 'Softmax' in func.label

        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing gradient update for {} ...'.format(func.label))
        print('{:20s}\t{}\t{}\t{}\t{}'.format(str(data[0].shape),
                                              xp.count_nonzero(data[0]),
                                              data[0].min(), data[0].max(),
                                              xp.linalg.norm(data[0])))

    def process_linear_data_grad(self, func, data):
        assert 'Data' in func.label and 'Linear' in func.label

        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing gradient update for {} ...'.format(func.label))

            for i in range(len(data)):
                print('{:20s}\t{}\t{}\t{}\tnorm={}'.format(
                    str(data[i].shape), xp.count_nonzero(data[i]),
                    data[i].min(), data[i].max(), xp.linalg.norm(data[i])))

    def process_convolution_2d_weight_grad(self, func, data):
        assert 'GradW' in func.label and 'Convolution' in func.label

        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing gradient update for {} ...'.format(func.label))

            for i in range(len(data)):
                print('{:20s}\t{}\t{}\t{}\tnorm={}'.format(
                    str(data[i].shape), xp.count_nonzero(data[i]),
                    data[i].min(), data[i].max(), xp.linalg.norm(data[i])))

    def process_deconvolution_2d(self, func, data):
        assert 'Deconvolution' in func.label

        xp = chainer.backend.get_array_module(data[0])

        # print out some results
        if self.verbose:
            print('Processing gradient update for {} ...'.format(func.label))

            for i in range(len(data)):
                print('{:20s}\t{}\t{}\t{}\tnorm={}'.format(
                    str(data[i].shape), xp.count_nonzero(data[i]),
                    data[i].min(), data[i].max(), xp.linalg.norm(data[i])))

    def process_linear_weight_grad(self, func, data):
        """ Process the gradient function w.r.t. weights """
        assert 'Weight' in func.label and 'Linear' in func.label

        if self.verbose:
            print('Processing gradient update for {} ...'.format(func.label))

        xp = chainer.backend.get_array_module(data[1])
        i = self.current_iteration
        self.history.append(
            [self.loss_scale, i, func.label,
             xp.count_nonzero(data[1])])

        # print out some results
        print('{:20s}\t{}\t{}\t{}\t{}'.format(str(data[1].shape),
                                              xp.count_nonzero(data[1]),
                                              data[1].min(), data[1].max(),
                                              xp.linalg.norm(data[1])))

    def export_history(self):
        return pd.DataFrame(self.history,
                            columns=[
                                'iter',
                                'label',
                                'index',
                                'size',
                                'nonzero',
                                'min',
                                'max',
                                'mean',
                                'std',
                                'norm',
                            ])
