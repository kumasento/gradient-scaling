from collections import defaultdict
import argparse

import numpy as np
import chainer
from chainer import iterators
from chainer.dataset import concat_examples
from chainer import functions as F

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.datasets import VOCSemanticSegmentationDataset

from chainercv.evaluations import eval_semantic_segmentation
from chainercv.experimental.links import PSPNetResNet101
from chainercv.experimental.links import PSPNetResNet50
from chainercv.links import DeepLabV3plusXception65
# from chainercv.links import SegNetBasic
from models.segnet.segnet_basic import SegNetBasic
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


models = {
    'pspnet_resnet50': (PSPNetResNet50, {}, 1),
    'pspnet_resnet101': (PSPNetResNet101, {}, 1),
    'segnet': (SegNetBasic, {}, 1),
    'deeplab_v3plus_xception65': (DeepLabV3plusXception65, {}, 1),
}


def recalculate_bn_statistics(model, batchsize, dtype='float32'):
  print('==> Recalculating BN statistics (batchsize={}) ...'.format(batchsize))
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


def setup(dataset, model, pretrained_model, batchsize, input_size):
  dataset_name = dataset
  if dataset_name == 'cityscapes':
    dataset = CityscapesSemanticSegmentationDataset(
        split='val', label_resolution='fine')
    label_names = cityscapes_semantic_segmentation_label_names
  elif dataset_name == 'ade20k':
    dataset = ADE20KSemanticSegmentationDataset(split='val')
    label_names = ade20k_semantic_segmentation_label_names
  elif dataset_name == 'camvid':
    dataset = CamVidDataset(split='test')
    label_names = camvid_label_names
  elif dataset_name == 'voc':
    dataset = VOCSemanticSegmentationDataset(split='val')
    label_names = voc_semantic_segmentation_label_names

  def eval_(out_values, rest_values):
    pred_labels, = out_values
    gt_labels, = rest_values

    result = eval_semantic_segmentation(pred_labels, gt_labels)

    for iu, label_name in zip(result['iou'], label_names):
      print('{:>23} : {:.4f}'.format(label_name, iu))
    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', result['miou']))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', result['mean_class_accuracy']))
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', result['pixel_accuracy']))

  cls, pretrained_models, default_batchsize = models[model]
  if pretrained_model is None:
    pretrained_model = pretrained_models.get(dataset_name, dataset_name)
  if input_size is None:
    input_size = None
  else:
    input_size = (input_size, input_size)

  kwargs = {
      'n_class': len(label_names),
      'pretrained_model': pretrained_model,
  }
  if model in ['pspnet_resnet50', 'pspnet_resnet101']:
    kwargs.update({'input_size': input_size})
  elif model == 'deeplab_v3plus_xception65':
    kwargs.update({'min_input_size': input_size})
  model = cls(**kwargs)

  if batchsize is None:
    batchsize = default_batchsize

  return dataset, eval_, model, batchsize


dtypes = {
    'float16': np.float16,
    'float32': np.float32,
    'mixed16': chainer.mixed16,
}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset', choices=('cityscapes', 'ade20k', 'camvid', 'voc'))
  parser.add_argument('--model', choices=sorted(models.keys()))
  parser.add_argument('--gpu', type=int, default=-1)
  parser.add_argument('--pretrained-model')
  parser.add_argument('--batchsize', type=int)
  parser.add_argument('--input-size', type=int, default=None)
  parser.add_argument('--dtype', type=str, default='float32')
  args = parser.parse_args()

  # setup dtype
  chainer.global_config.dtype = dtypes[args.dtype]

  dataset, eval_, model, batchsize = setup(
      args.dataset, args.model, args.pretrained_model,
      args.batchsize, args.input_size)

  model.dtype = dtypes[args.dtype]

  if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

  if args.model == 'segnet':
    recalculate_bn_statistics(model, 12, dtype=args.dtype)

  iterator = iterators.SerialIterator(
      dataset, batchsize, repeat=False, shuffle=False)

  in_values, out_values, rest_values = apply_to_iterator(
      model.predict, iterator, hook=ProgressHook(len(dataset)))
  # Delete an iterator of images to save memory usage.
  del in_values

  eval_(out_values, rest_values)


if __name__ == '__main__':
  main()
