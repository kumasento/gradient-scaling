import os
import argparse
import sys
import copy
import numpy as np
import cupy as cp
import random
from contextlib import ExitStack

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv import transforms

from models.ssd import GradientScaling
from models.ssd import multibox_loss
from models.ssd import random_crop_with_bbox_constraints
from models.ssd import random_distort
from models.ssd import resize_with_random_interpolation
from models.ssd.ssd_vgg16 import SSD300, SSD512

import chainerlp.extensions
from chainerlp.hooks import AdaLossMonitor

# from chainerlp.ada_loss.transforms import *

# AdaLoss support
from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer_impl.ada_loss_transforms import AdaLossTransformLinear
from ada_loss.chainer_impl.transforms import AdaLossTransformConvolution2D
from ada_loss.chainer_impl.ada_loss_recorder import AdaLossRecorder
from ada_loss.chainer_impl.sanity_checker import SanityChecker
from ada_loss.profiler import Profiler

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
import cv2

cv2.setNumThreads(0)
chainer.global_config.cv_resize_backend = "cv2"

chainer.global_config.warn_nondeterministic = True

np.random.seed(0)
cp.random.seed(0)
random.seed(0)
chainer.global_config.cudnn_deterministic = True


class MultiboxTrainChain(chainer.Chain):
    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def forward(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k
        )
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {"loss": loss, "loss/loc": loc_loss, "loss/conf": conf_loss}, self
        )

        return loss


# data types
dtypes = {
    "float16": np.float16,
    "float32": np.float32,
    "mixed16": chainer.mixed16,
}


class Transform(object):
    def __init__(self, coder, size, mean, dtype=None):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean
        self.dtype = dtype

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True
            )
            bbox = transforms.translate_bbox(
                bbox, y_offset=param["y_offset"], x_offset=param["x_offset"]
            )

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox,
            y_slice=param["y_slice"],
            x_slice=param["x_slice"],
            allow_outside_center=False,
            return_param=True,
        )
        label = label[param["index"]]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params["x_flip"]
        )

        # Preparation for SSD network
        img -= self.mean

        mb_loc, mb_label = self.coder.encode(bbox, label)

        dtype = chainer.get_dtype(self.dtype)
        if img.dtype != dtype:
            img = img.astype(dtype)

        return img, mb_loc, mb_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ssd300", "ssd512"), default="ssd300")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--test-batchsize", type=int, default=16)
    parser.add_argument("--iteration", type=int, default=120000)
    parser.add_argument("--step", type=int, nargs="*", default=[80000, 100000])
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--out", default="result")
    parser.add_argument("--resume")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtypes.keys(),
        default="float32",
        help="Select the data type of the model",
    )
    parser.add_argument(
        "--model-dir", default=None, type=str, help="Where to store models"
    )
    parser.add_argument(
        "--dataset-dir", default=None, type=str, help="Where to store datasets"
    )
    parser.add_argument(
        "--dynamic-interval",
        default=None,
        type=int,
        help="Interval for dynamic loss scaling",
    )
    parser.add_argument(
        "--init-scale", default=1, type=float, help="Initial scale for ada loss"
    )
    parser.add_argument(
        "--loss-scale-method",
        default="approx_range",
        type=str,
        help="Method for adaptive loss scaling",
    )
    parser.add_argument(
        "--scale-upper-bound",
        default=32800,
        type=float,
        help="Hard upper bound for each scale factor",
    )
    parser.add_argument(
        "--accum-upper-bound",
        default=32800,
        type=float,
        help="Accumulated upper bound for all scale factors",
    )
    parser.add_argument(
        "--update-per-n-iteration",
        default=100,
        type=int,
        help="Update the loss scale value per n iteration",
    )
    parser.add_argument(
        "--snapshot-per-n-iteration",
        default=10000,
        type=int,
        help="The frequency of taking snapshots",
    )
    parser.add_argument("--n-uf", default=1e-3, type=float)
    parser.add_argument("--nosanity-check", default=False, action="store_true")
    parser.add_argument("--nouse-fp32-update", default=False, action="store_true")
    parser.add_argument("--profiling", default=False, action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Verbose output"
    )
    args = parser.parse_args()

    # Setting data types
    if args.dtype != "float32":
        chainer.global_config.use_cudnn = "never"
    chainer.global_config.dtype = dtypes[args.dtype]
    print("==> Setting the data type to {}".format(args.dtype))

    # Initialize model
    if args.model == "ssd300":
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names), pretrained_model="imagenet"
        )
    elif args.model == "ssd512":
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names), pretrained_model="imagenet"
        )

    model.use_preset("evaluate")

    # Apply adaptive loss scaling
    recorder = AdaLossRecorder(sample_per_n_iter=100)
    profiler = Profiler()
    sanity_checker = (
        SanityChecker(check_per_n_iter=100) if not args.nosanity_check else None
    )
    # Update the model to support AdaLoss
    # TODO: refactorize
    model_ = AdaLossScaled(
        model,
        init_scale=args.init_scale,
        cfg={
            "loss_scale_method": args.loss_scale_method,
            "scale_upper_bound": args.scale_upper_bound,
            "accum_upper_bound": args.accum_upper_bound,
            "update_per_n_iteration": args.update_per_n_iteration,
            "recorder": recorder,
            "profiler": profiler,
            "sanity_checker": sanity_checker,
            "n_uf_threshold": args.n_uf,
        },
        transforms=[AdaLossTransformLinear(), AdaLossTransformConvolution2D(),],
        verbose=args.verbose,
    )

    # Finalize the model
    train_chain = MultiboxTrainChain(model_)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cp.random.seed(0)

        # NOTE: we have to transfer modules explicitly to GPU
        model.coder.to_gpu()
        model.extractor.to_gpu()
        model.multibox.to_gpu()

    # Prepare dataset
    if args.model_dir is not None:
        chainer.dataset.set_dataset_root(args.model_dir)
    train = TransformDataset(
        ConcatenatedDataset(
            VOCBboxDataset(year="2007", split="trainval"),
            VOCBboxDataset(year="2012", split="trainval"),
        ),
        Transform(model.coder, model.insize, model.mean, dtype=dtypes[args.dtype]),
    )
    # train_iter = chainer.iterators.MultiprocessIterator(
    #     train, args.batchsize) # , n_processes=8, n_prefetch=2)
    train_iter = chainer.iterators.MultithreadIterator(train, args.batchsize)
    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    test = VOCBboxDataset(
        year="2007", split="test", use_difficult=True, return_difficult=True
    )
    test_iter = chainer.iterators.SerialIterator(
        test, args.test_batchsize, repeat=False, shuffle=False
    )

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    if args.dtype == "mixed16":
        if not args.nouse_fp32_update:
            print("==> Using FP32 update for dtype=mixed16")
            optimizer.use_fp32_update()  # by default use fp32 update

        # HACK: support skipping update by existing loss scaling functionality
        if args.dynamic_interval is not None:
            optimizer.loss_scaling(interval=args.dynamic_interval, scale=None)
        else:
            optimizer.loss_scaling(interval=float("inf"), scale=None)
            optimizer._loss_scale_max = 1.0  # to prevent actual loss scaling

    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == "b":
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, "iteration"), args.out)
    trainer.extend(
        extensions.ExponentialShift("lr", 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger(args.step, "iteration"),
    )

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True, label_names=voc_bbox_label_names
        ),
        trigger=triggers.ManualScheduleTrigger(
            args.step + [args.iteration], "iteration"
        ),
    )

    log_interval = 10, "iteration"
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.observe_value(
            "loss_scale",
            lambda trainer: trainer.updater.get_optimizer("main")._loss_scale,
        ),
        trigger=log_interval,
    )

    metrics = [
        "epoch",
        "iteration",
        "lr",
        "main/loss",
        "main/loss/loc",
        "main/loss/conf",
        "validation/main/map",
    ]
    if args.dynamic_interval is not None:
        metrics.insert(2, "loss_scale")
    trainer.extend(extensions.PrintReport(metrics), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        extensions.snapshot(),
        trigger=triggers.ManualScheduleTrigger(
            args.step + [args.iteration], "iteration"
        ),
    )
    trainer.extend(
        extensions.snapshot_object(model, "model_iter_{.updater.iteration}"),
        trigger=(args.iteration, "iteration"),
    )

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    hook = AdaLossMonitor(
        sample_per_n_iter=100, verbose=args.verbose, includes=["Grad", "Deconvolution"]
    )
    recorder.trainer = trainer
    hook.trainer = trainer

    with ExitStack() as stack:
        stack.enter_context(hook)
        trainer.run()

    recorder.export().to_csv(os.path.join(args.out, "loss_scale.csv"))
    profiler.export().to_csv(os.path.join(args.out, "profile.csv"))
    if sanity_checker:
        sanity_checker.export().to_csv(os.path.join(args.out, "sanity_check.csv"))
    hook.export_history().to_csv(os.path.join(args.out, "grad_stats.csv"))


if __name__ == "__main__":
    main()
