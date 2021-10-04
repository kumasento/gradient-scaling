""" Utility functions for analyzing ImageNet results. """

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import chainer

from chainerlp import notebook_utils


def load_training_results(out_dir):
    """ Load training results from the given directory. """
    assert os.path.isdir(out_dir)

    ldf = notebook_utils.load_train_log(train_dir=out_dir)
    if os.path.isfile(os.path.join(out_dir, "grad_stats.csv")):
        gdf = pd.read_csv(os.path.join(out_dir, "grad_stats.csv"), index_col=0)
        sdf = pd.read_csv(os.path.join(out_dir, "loss_scale.csv"), index_col=0)

        return gdf, sdf, ldf

    return None, None, ldf


def plot_grad_nonzero(gdf, iters, label="ReLUGrad2"):
    """ Plot the number of nonzeros across all layers at given iterations. """
    fig, ax = plt.subplots()

    # filter given function labels
    gdf_ = gdf[gdf["label"] == label]
    # collect data at all iterations
    dfs = []
    for it in iters:
        df = gdf_[gdf_["iter"] == it]
        dfs.append(df)

        n_layer = len(df)
        ax.plot(np.arange(n_layer)[::-1], df["nonzero"] / df["size"] * 100)
