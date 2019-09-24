""" Record the loss scale value.
    Pass it as one ada_loss_cfg field.
"""
import pandas as pd


class AdaLossRecorder(object):
    """ """

    def __init__(self, sample_per_n_iter=None):
        """ """
        self.data = []

        self.sample_per_n_iter = sample_per_n_iter

    def setup(self, trainer):
        """ assign trainer """
        self.trainer = trainer

    @property
    def current_iteration(self):
        return self.trainer.updater.iteration

    def is_recording(self):
        """ """
        if self.sample_per_n_iter is None:
            return True

        return self.current_iteration % self.sample_per_n_iter == 0

    def record(self, key, val, label=None):
        """ Store the loss scale value """
        if self.is_recording():
            self.data.append([
                self.current_iteration,
                label,
                key,
                val,
            ])

    def export(self):
        """ Export to format that can be further processed """
        return pd.DataFrame(self.data,
                            columns=[
                                'iter',
                                'label',
                                'key',
                                'val',
                            ])
