""" Train model on CIFAR """

import os
import argparse
import chainer
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

# training utilities
import utils

parser = argparse.ArgumentParser(prog="Train ResNet on CIFAR")
# Model parameters
parser.add_argument("--n_layer", type=int, default=20, help="Number of ResNet layers")
parser.add_argument(
    "--n_class", type=int, default=10, help="Number of classes (select 10 or 100)"
)
# AdaLoss parameters
parser.add_argument("--method", type=str, default="approx_range", help="AdaLoss method")
parser.add_argument(
    "--loss_scale", type=float, default=1, help="Initial loss scale value"
)
parser.add_argument(
    "--update_per_n_iteration",
    type=int,
    default=1,
    help="The frequency of updating loss scale.",
)
# Training parameters
parser.add_argument("--out", type=str, default=None, help="Where to store results.")
parser.add_argument(
    "--manual_seed",
    type=int,
    default=0,
    help="Random seed for training and model initialization (default: 0)",
)
parser.add_argument("--train_batch", type=int, default=128)
# Device
parser.add_argument("--gpu", type=int, default=-1, help="GPU ID")
parser.add_argument("--lr", type=float, default=0.1)
args = parser.parse_args()


def main():
    """ Run training """
    kwargs = {}
    if args.n_layer >= 110:  # additional config
        kwargs["warmup_attr_ratio"] = 0.1
        kwargs["warmup_n_epoch"] = 5

    if args.update_per_n_iteration != 1:
        args.out += "-freq_{}".format(args.update_per_n_iteration)

    gdf, sdf, ldf = utils.train(
        args.n_layer,
        n_class=args.n_class,
        method=args.method,
        init_scale=args.loss_scale,
        device=args.gpu,
        update_per_n_iteration=args.update_per_n_iteration,
        manual_seed=args.manual_seed,
        train_batch=args.train_batch,
        learnrate=args.lr,
        **kwargs
    )

    # Plot results
    fig = utils.plot(
        gdf,
        sdf,
        ldf,
        title="Train ResNet-{n_layer} on CIFAR-{n_class} by {method} (scale={loss_scale}, seed={manual_seed})".format(
            n_layer=args.n_layer,
            n_class=args.n_class,
            method=args.method,
            loss_scale=args.loss_scale,
            manual_seed=args.manual_seed,
        ),
    )
    # Save results
    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)

        gdf.to_csv(os.path.join(args.out, "grad_stats.csv"))
        sdf.to_csv(os.path.join(args.out, "loss_scale.csv"))
        ldf.to_csv(os.path.join(args.out, "train_logs.csv"))
        fig.savefig(os.path.join(args.out, "summary.pdf"))


if __name__ == "__main__":
    main()
