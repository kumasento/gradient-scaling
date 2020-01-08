""" Test adaptive loss scaling for segnet """

import unittest

import chainer
from chainer import testing
from chainer import functions as F
from chainer.dataset import concat_examples

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset

from ada_loss.chainer_impl.ada_loss_scaled import AdaLossScaled
from ada_loss.chainer_impl.ada_loss_transforms import AdaLossTransformLinear
from ada_loss.chainer_impl.transforms import AdaLossTransformConvolution2D
from ada_loss.chainer_impl.transforms import AdaLossTransformBatchNormalization

from models.segnet.segnet_basic import SegNetBasic


class TestSegNetBasic(unittest.TestCase):

  def test_ada_loss_scaled(self):
    dtype = 'float16'

    # change the data type before buidling the model
    chainer.global_config.dtype = dtype
    model = SegNetBasic(n_class=len(camvid_label_names), dtype=dtype)
    model = AdaLossScaled(
        model,
        transforms=[
            AdaLossTransformLinear(),
            AdaLossTransformConvolution2D(),
            AdaLossTransformBatchNormalization(),
        ],
        verbose=True)
    print(model)

    # evaluate
    # train = CamVidDataset(split='train')
    # it = chainer.iterators.SerialIterator(
    #     train, 1, repeat=False, shuffle=False)
    # batch = next(it)
    # imgs, _ = concat_examples(batch)
    # x = F.cast(model.xp.array(imgs), dtype)
    # golden = model(x)
    # print(golden)


testing.run_module(__name__, __file__)
