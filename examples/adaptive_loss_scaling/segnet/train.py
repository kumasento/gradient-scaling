import argparse
from collections import defaultdict
import os
from contextlib import ExitStack

import chainer
import numpy as np

from chainer.dataset import concat_examples
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import functions as F

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.extensions import SemanticSegmentationEvaluator
# from chainercv.links import PixelwiseSoftmaxClassifier
# from chainercv.links import SegNetBasic

import chainerlp.extensions
from chainerlp.hooks import AdaLossMonitor

from ada_loss.chainer_impl.ada_loss import AdaLossChainer
from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer_impl.ada_loss_transforms import AdaLossTransformLinear
from ada_loss.chainer_impl.transforms import AdaLossTransformConvolution2D
from ada_loss.chainer_impl.transforms import AdaLossTransformBatchNormalization
from ada_loss.chainer_impl.ada_loss_recorder import AdaLossRecorder

# Change where to import SegNet
from models.segnet.segnet_basic import SegNetBasic
from models.segnet.pixelwise_softmax_classifier import PixelwiseSoftmaxClassifier 

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def recalculate_bn_statistics(model, batchsize, dtype='float32'):
    train = CamVidDataset(split='train')
    it = chainer.iterators.SerialIterator(
        train, batchsize, repeat=False, shuffle=False)
    bn_avg_mean = defaultdict(np.float32)
    bn_avg_var = defaultdict(np.float32)

    if dtype == 'mixed16':
        dtype = 'float16'

    n_iter = 0
    for batch in it:
        imgs, _ = concat_examples(batch)

        model(F.cast(model.xp.array(imgs), dtype))
        for name, link in model.namedlinks():
            if name.endswith('_bn'):
                bn_avg_mean[name] += link.avg_mean
                bn_avg_var[name] += link.avg_var
        n_iter += 1

    for name, link in model.namedlinks():
        if name.endswith('_bn'):
            link.avg_mean = bn_avg_mean[name] / n_iter
            link.avg_var = bn_avg_var[name] / n_iter

    return model


def transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    
    dtype = chainer.global_config.dtype
    if dtype != 'float32':
        img = img.astype(dtype)

    return img, label

dtypes = {
    'float16': np.float16,
    'float32': np.float32,
    'mixed16': chainer.mixed16,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--class-weight', type=str, default='class_weight.npy')
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--iter', type=int, default=16000)
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--init-scale', type=float, default=1.0)
    parser.add_argument('--dynamic-interval', type=int, default=None)
    parser.add_argument('--loss-scale-method', type=str, default='approx_range')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    # Data type
    print('==> Using data type: {}'.format(args.dtype))
    chainer.global_config.dtype = dtypes[args.dtype]

    # Triggers
    log_trigger = (50, 'iteration')
    validation_trigger = (2000, 'iteration')
    end_trigger = (args.iter, 'iteration')

    # Dataset
    train = CamVidDataset(split='train')
    train = TransformDataset(train, transform)
    val = CamVidDataset(split='val')

    # Iterator
    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = iterators.MultiprocessIterator(
        val, args.batchsize, shuffle=False, repeat=False)

    # Model
    class_weight = np.load(args.class_weight)
    model = SegNetBasic(n_class=len(camvid_label_names), dtype=dtypes[args.dtype])
    
    # adaptive loss scaling
    recorder = AdaLossRecorder(sample_per_n_iter=100)
    model = AdaLossScaled(
        model,
        init_scale=args.init_scale,
        transforms=[
            AdaLossTransformLinear(),
            AdaLossTransformConvolution2D(),
            AdaLossTransformBatchNormalization(),
        ],
        cfg={
          'loss_scale_method': args.loss_scale_method,
          'scale_upper_bound': 65504,
          'accum_upper_bound': 65504,
          'update_per_n_iteration': 100,
          'recorder': recorder,
          'n_uf_threshold': 1e-3,
        },
        verbose=args.verbose)

    model = PixelwiseSoftmaxClassifier(
        model, class_weight=class_weight)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))

    if args.dtype == 'mixed16':
        optimizer.use_fp32_update()

    # Updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    if args.dtype == 'mixed16':
        if args.dynamic_interval is not None:
            print('==> Using dynamic loss scaling (interval={}) ...'.format(
                args.dynamic_interval))
            optimizer.loss_scaling(interval=args.dynamic_interval, scale=None)
        else:
            if args.loss_scale_method == 'approx_range':
                print('==> Using adaptive loss scaling ...')
            else:
                print('==> Using default fixed loss scaling (scale={}) ...'.format(
                    args.init_scale))

            optimizer.loss_scaling(interval=float('inf'), scale=None)
            optimizer._loss_scale_max = 1.0 # to prevent actual loss scaling

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=args.out)

    # warmup_attr_ratio = 0.1 if args.dtype != 'float32' else None
    # # NOTE: this is confusing but it means n_iter
    # warmup_n_epoch = 1000 if args.dtype != 'float32' else None
    # lr_shift = chainerlp.extensions.ExponentialShift(
    #     'lr',
    #     0.1,
    #     init=0.1 * warmup_attr_ratio,
    #     warmup_attr_ratio=warmup_attr_ratio,
    #     warmup_n_epoch=warmup_n_epoch)
    # trainer.extend(lr_shift, trigger=(1, 'iteration'))

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.observe_value(
        'loss_scale',
        lambda trainer: trainer.updater.get_optimizer('main')._loss_scale),
                   trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss'], x_key='iteration',
            file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['validation/main/miou'], x_key='iteration',
            file_name='miou.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr', 'loss_scale',
         'main/loss', 'validation/main/miou',
         'validation/main/mean_class_accuracy',
         'validation/main/pixel_accuracy']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        SemanticSegmentationEvaluator(
            val_iter, model.predictor,
            camvid_label_names),
        trigger=validation_trigger)

    hook = AdaLossMonitor(sample_per_n_iter=100,
                          verbose=args.verbose,
                          includes=['Grad', 'Deconvolution'])
    recorder.trainer = trainer
    hook.trainer = trainer

    with ExitStack() as stack:
        stack.enter_context(hook)
        trainer.run()

    chainer.serializers.save_npz(
        os.path.join(args.out, 'snapshot_model.npz'),
        recalculate_bn_statistics(model.predictor, 24, dtype=args.dtype))
    recorder.export().to_csv(os.path.join(args.out, 'loss_scale.csv'))

if __name__ == '__main__':
    main()
