import os
import sys
import argparse
import multiprocessing
import numpy as np
from contextlib import ExitStack

import chainer
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

import chainermn

from chainercv.chainer_experimental.datasets.sliceable \
    import ConcatenatedDataset
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator

from models.ssd import GradientScaling
from models.ssd import multibox_loss
from models.ssd.ssd_vgg16 import SSD300, SSD512

from train import Transform

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
# chainer.global_config.debug = True

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
import cv2
cv2.setNumThreads(0)


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3, comm=None):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k
        self.comm = comm

    def forward(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(mb_locs, mb_confs, gt_mb_locs,
                                            gt_mb_labels, self.k, self.comm)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {
                'loss': loss,
                'loss/loc': loc_loss,
                'loss/conf': conf_loss
            }, self)

        return loss


# data types
dtypes = {
    'float16': np.float16,
    'float32': np.float32,
    'mixed16': chainer.mixed16,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        choices=('ssd300', 'ssd512'),
                        default='ssd300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--np', type=int, default=8)
    parser.add_argument('--test-batchsize', type=int, default=16)
    parser.add_argument('--iteration', type=int, default=120000)
    parser.add_argument('--step', type=int, nargs='*', default=[80000, 100000])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--dtype',
                        type=str,
                        choices=dtypes.keys(),
                        default='float32',
                        help='Select the data type of the model')
    parser.add_argument('--model-dir',
                        default=None,
                        type=str,
                        help='Where to store models')
    parser.add_argument('--dataset-dir',
                        default=None,
                        type=str,
                        help='Where to store datasets')
    parser.add_argument('--dynamic-interval',
                        default=None,
                        type=int,
                        help='Interval for dynamic loss scaling')
    parser.add_argument('--init-scale',
                        default=1,
                        type=float,
                        help='Initial scale for ada loss')
    parser.add_argument('--loss-scale-method',
                        default='approx_range',
                        type=str,
                        help='Method for adaptive loss scaling')
    parser.add_argument('--scale-upper-bound',
                        default=16,
                        type=float,
                        help='Hard upper bound for each scale factor')
    parser.add_argument('--accum-upper-bound',
                        default=1024,
                        type=float,
                        help='Accumulated upper bound for all scale factors')
    parser.add_argument('--update-per-n-iteration',
                        default=1,
                        type=int,
                        help='Update the loss scale value per n iteration')
    parser.add_argument('--snapshot-per-n-iteration',
                        default=10000,
                        type=int,
                        help='The frequency of taking snapshots')
    parser.add_argument('--n-uf', default=1e-3, type=float)
    parser.add_argument('--nosanity-check', default=False, action='store_true')
    parser.add_argument('--nouse-fp32-update', default=False, action='store_true')
    parser.add_argument('--profiling', default=False, action='store_true')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Verbose output')
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank

    # Set up workspace
    # 12 GB GPU RAM for workspace
    chainer.cuda.set_max_workspace_size(16 * 1024 * 1024 * 1024)
    chainer.global_config.cv_resize_backend = 'cv2'

    # Setup the data type
    # when initializing models as follows, their data types will be casted.
    # Weethave to forbid the usage of cudnn
    if args.dtype != 'float32':
        chainer.global_config.use_cudnn = 'never'
    chainer.global_config.dtype = dtypes[args.dtype]
    print('==> Setting the data type to {}'.format(args.dtype))

    if args.model_dir is not None:
        chainer.dataset.set_dataset_root(args.model_dir)
    if args.model == 'ssd300':
        model = SSD300(n_fg_class=len(voc_bbox_label_names),
                       pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(n_fg_class=len(voc_bbox_label_names),
                       pretrained_model='imagenet')

    model.use_preset('evaluate')

    ######################################
    # Setup model
    #######################################
    # Apply ada loss transform
    recorder = AdaLossRecorder(sample_per_n_iter=100)
    profiler = Profiler()
    sanity_checker = SanityChecker(check_per_n_iter=100) if not args.nosanity_check else None
    # Update the model to support AdaLoss
    # TODO: refactorize
    model_ = AdaLossScaled(
        model,
        init_scale=args.init_scale,
        cfg={
            'loss_scale_method': args.loss_scale_method,
            'scale_upper_bound': args.scale_upper_bound,
            'accum_upper_bound': args.accum_upper_bound,
            'update_per_n_iteration': args.update_per_n_iteration,
            'recorder': recorder,
            'profiler': profiler,
            'sanity_checker': sanity_checker,
            'n_uf_threshold': args.n_uf,
            # 'power_of_two': False,
        },
        transforms=[
            AdaLossTransformLinear(),
            AdaLossTransformConvolution2D(),
        ],
        verbose=args.verbose)

    if comm.rank == 0:
        print(model)

    train_chain = MultiboxTrainChain(model_, comm=comm)
    chainer.cuda.get_device_from_id(device).use()

    # to GPU
    model.coder.to_gpu()
    model.extractor.to_gpu()
    model.multibox.to_gpu()

    shared_mem = 100 * 1000 * 1000 * 4

    if args.dataset_dir is not None:
        chainer.dataset.set_dataset_root(args.dataset_dir)
    train = TransformDataset(
        ConcatenatedDataset(VOCBboxDataset(year='2007', split='trainval'),
                            VOCBboxDataset(year='2012', split='trainval')),
        ('img', 'mb_loc', 'mb_label'),
        Transform(model.coder,
                  model.insize,
                  model.mean,
                  dtype=dtypes[args.dtype]))

    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.MultiprocessIterator(train,
                                                        args.batchsize //
                                                        comm.size,
                                                        n_processes=8,
                                                        n_prefetch=2,
                                                        shared_mem=shared_mem)

    if comm.rank == 0:  # NOTE: only performed on the first device
        test = VOCBboxDataset(year='2007',
                              split='test',
                              use_difficult=True,
                              return_difficult=True)
        test_iter = chainer.iterators.SerialIterator(test,
                                                     args.test_batchsize,
                                                     repeat=False,
                                                     shuffle=False)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    if args.dtype == 'mixed16':
        if not args.nouse_fp32_update:
            print('==> Using FP32 update for dtype=mixed16')
            optimizer.use_fp32_update()  # by default use fp32 update

        # HACK: support skipping update by existing loss scaling functionality
        if args.dynamic_interval is not None:
            optimizer.loss_scaling(interval=args.dynamic_interval, scale=None)
        else:
            optimizer.loss_scaling(interval=float('inf'), scale=None)
            optimizer._loss_scale_max = 1.0 # to prevent actual loss scaling



    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(train_iter,
                                                optimizer,
                                                device=device)
    # if args.dtype == 'mixed16':
    #     updater.loss_scale = 8
    iteration_interval = (args.iteration, 'iteration')

    trainer = training.Trainer(updater, iteration_interval, args.out)
    # trainer.extend(extensions.ExponentialShift('lr', 0.1, init=args.lr),
    #                trigger=triggers.ManualScheduleTrigger(
    #                    args.step, 'iteration'))
    if args.batchsize != 32:
      warmup_attr_ratio = 0.1
      # NOTE: this is confusing but it means n_iter
      warmup_n_epoch = 1000
      lr_shift = chainerlp.extensions.ExponentialShift(
          'lr',
          0.1,
          init=args.lr * warmup_attr_ratio,
          warmup_attr_ratio=warmup_attr_ratio,
          warmup_n_epoch=warmup_n_epoch,
          schedule=args.step)
      trainer.extend(lr_shift, trigger=(1, 'iteration'))

    if comm.rank == 0:
        if not args.profiling:
            trainer.extend(DetectionVOCEvaluator(test_iter,
                                                model,
                                                use_07_metric=True,
                                                label_names=voc_bbox_label_names),
                        trigger=triggers.ManualScheduleTrigger(
                            args.step + [args.iteration], 'iteration'))

        log_interval = 10, 'iteration'
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.observe_value(
            'loss_scale',
            lambda trainer: trainer.updater.get_optimizer('main')._loss_scale),
                       trigger=log_interval)

        metrics = [
            'epoch', 'iteration', 'lr', 'main/loss',
            'main/loss/loc', 'main/loss/conf', 'validation/main/map'
        ]
        if args.dynamic_interval is not None:
            metrics.insert(2, 'loss_scale')

        trainer.extend(extensions.PrintReport(metrics), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        trainer.extend(extensions.snapshot(),
                       trigger=(args.snapshot_per_n_iteration, 'iteration'))
        trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}'),
                       trigger=(args.iteration, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    hook = AdaLossMonitor(sample_per_n_iter=100,
                          verbose=args.verbose,
                          includes=['Grad', 'Deconvolution'])
    recorder.trainer = trainer
    hook.trainer = trainer

    with ExitStack() as stack:
        if comm.rank == 0:
            stack.enter_context(hook)
        trainer.run()

    # store recorded results
    if comm.rank == 0:  # NOTE: only export in the first rank
        recorder.export().to_csv(os.path.join(args.out, 'loss_scale.csv'))
        profiler.export().to_csv(os.path.join(args.out, 'profile.csv'))
        if sanity_checker:
            sanity_checker.export().to_csv(os.path.join(args.out, 'sanity_check.csv'))
        hook.export_history().to_csv(os.path.join(args.out, 'grad_stats.csv'))



if __name__ == '__main__':
    main()
