""" Hook to log activations """

import os
import pickle
from collections import OrderedDict

import chainer
import chainer.links as L
from chainer import backend
from chainer import link_hook
from chainer import function_hook


class ActStatLinkHook(link_hook.LinkHook):
    """ Definition of the hook that records the statistics of activation. """

    name = "ActStatFunctionHook"

    def __init__(self):
        """ CTOR """
        pass

    def forward_postprocess(self, args):
        """ Where the logging happens """
        assert isinstance(args, link_hook._ForwardPostprocessCallbackArgs)

        link, out = args.link, args.out
        print(
            link.name,
            type(link),
            out.shape,
            out.data.min(),
            out.data.max(),
            out.data.mean(),
            out.data.std(),
        )


class ActStatFuncHook(function_hook.FunctionHook):
    """ Definition of the hook that records the statistics of activation. """

    name = "ActStatFunctionHook"

    def __init__(
        self, trainer=None, snapshot_dir=None, snapshot_prefix="act_stat", excludes=None
    ):
        """ CTOR """
        self.trainer = trainer
        self.init_states()

        if excludes is None:
            self.excludes = []
        else:
            self.excludes = excludes

        self.n_iter = 0
        self.snapshot_dir = snapshot_dir if snapshot_dir is not None else "."
        self.snapshot_prefix = snapshot_prefix

    def init_states(self):
        """ Initialize the states to be recorded """
        self.call_history = OrderedDict()  # where to store the stats to be recorded
        self.id = 0  # ID of function nodes

    def forward_postprocess(self, function, in_data):
        """ Forward computation """
        self.process(function, in_data)

    def backward_postprocess(self, function, in_data, out_data):
        """ Backward computation """
        self.process(function, in_data, out_data=out_data)

    @property
    def curr_iter(self):
        """ Get the current iteration. """
        return self.trainer.updater.iteration

    def process(self, func, in_data, out_data=None):
        """ An uniform interface to process collected activations """
        assert isinstance(func, chainer.FunctionNode)

        for exclude in self.excludes:
            if exclude in func.label:  # excluded for further processing
                return

        is_backward = out_data is not None
        # print('==> {:6d} {:10s} {:30s}: num. input: {} num. output: {}'.format(
        #     self.curr_iter, 'BACKWARD' if is_backward else 'FORWARD',
        #     func.label, len(in_data), len(out_data) if is_backward else 0))

        if is_backward:
            self.save_stats(func, out_stats=self.process_data(out_data))
        else:
            self.save_stats(func, in_stats=self.process_data(in_data))

    def process_data(self, data_tuple):
        """ Process a tuple of ND arrays. """
        stats = OrderedDict()

        for i, data in enumerate(data_tuple):
            if data is None:  # data could be None
                continue
            stats_ = self.calc_stats(data)
            stats[i] = stats_

        return stats

    def calc_stats(self, data):
        """ Collect the statistics we want to get from an ND array.
            For now, we look at the min, max, mean, and stddev
        """
        # convert data type before computing stddev
        data_ = data.astype("float32")
        return data.min(), data.max(), data.mean(), data_.std()

    def save_stats(self, func, in_stats=None, out_stats=None):
        """ Save stats to call_history. """
        # NOTE: we won't be able to trigger the snapshot action when
        # a trainer is not at present.
        # print('Saving {} {} {} in_stats={} out_stats={}'.format(
        #     self.id, id(func), func.label, in_stats, out_stats))
        if self.trainer and self.n_iter != self.curr_iter:
            self.snapshot()

        # calculate and update the id of the current function
        if in_stats is not None:  # in the forward pass
            self.id += 1

        if id(func) not in self.call_history:
            self.call_history[id(func)] = [self.id, func.label, None, None]
        if in_stats is not None:
            self.call_history[id(func)][2] = in_stats
        if out_stats is not None:
            self.call_history[id(func)][3] = out_stats

    def snapshot(self):
        """ Save the call_history and clear it up. """
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.n_iter += 1

        fp = os.path.join(
            self.snapshot_dir,
            "{}_iter_{}.pkl".format(self.snapshot_prefix, self.n_iter),
        )
        print("==> Saving pickled call history to {} ...".format(fp))
        with open(fp, "wb") as f:
            pickle.dump(self.call_history, f)

        # reset states
        self.init_states()
