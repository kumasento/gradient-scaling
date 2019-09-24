""" Check whether the application of adaptive loss scale is correct or not. """

import chainer
import pandas as pd

class SanityChecker(object):
    """ Sanity checker """

    def __init__(self, check_per_n_iter=1000):
        self.check_per_n_iter = check_per_n_iter
        self.history = []
        self.counter = 0
        self.curr_iter = 0

    def check(self, gy, W, g1, g2, g3, loss_scale, n_uf, curr_iter):
        """ Check whether loss scale can help reliefing the underflow. """
        xp = chainer.backend.get_array_module(g1)
        if self.curr_iter != curr_iter:
            self.curr_iter = curr_iter
            self.counter = 0

        if (xp.isnan(g1.array).any() or
            xp.isnan(g2.array).any() or
            xp.isnan(gy.array).any() or
            xp.isnan(W.array).any()):
            return
        nnz1 = xp.count_nonzero(g1.array)
        nnz2 = xp.count_nonzero(g2.array) # fp16
        nnz3 = xp.count_nonzero(g3.array) # fp32

        nuf1 = nnz1 - nnz3
        nuf2 = nnz2 - nnz3

        self.history.append([
            self.curr_iter, self.counter, loss_scale.item(),
            nnz1.item(), nnz2.item(), nnz3.item(), 
            (nuf1 / g3.size * 100).item(),
            (nuf2 / g3.size * 100).item(),
        ])
        # print(self.history[-1])
        self.counter += 1

        # variance calculation
        # gy = gy.array.astype('float32')
        # W = W.array.astype('float32')
        # g2 = g2.array.astype('float32')
        # mu1, sigma1 = gy.mean(), gy.std()
        # mu2, sigma2 = W.mean(), W.std()
        # mu3, sigma3 = g2.mean(), g2.std()
        # mu3_ = mu1 * mu2
        # sigma3_ = xp.sqrt(((sigma1**2 + mu1**2) * (sigma2**2 + mu2**2) - (mu3_)**2))
        # print('mu1: {} mu2: {} sigma1: {} sigma2: {}'.format(mu1, mu2, sigma1, sigma2))
        # print('mu3: {} {} sigma3: {} {}'.format(mu3, mu3_, sigma3, sigma3_))

        # print('scale: {} NNZ scaled: {} ({:.4f}%) base: {} ({:.4f}%) float32: {} ({:.4f}%) n_uf: {:.4f}%'.format(
        #     loss_scale.item(),
        #     nnz1.item(),
        #     (100 - nnz1 / g1.size * 100).item(),
        #     nnz2.item(),
        #     (100 - nnz2 / g2.size * 100).item(),
        #     nnz3.item(),
        #     (100 - nnz3 / g3.size * 100).item(),
        #     n_uf * 100))

    def export(self):
        return pd.DataFrame(self.history, columns=[
            'iter', 'id', 'loss_scale', 'nnz_ls', 'nnz_fp16', 'nnz_fp32',
            'nuf_ls', 'nuf_fp16'])
        
