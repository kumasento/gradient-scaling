""" The extension of SSD model """

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from models.ssd import Multibox
from models.ssd import Normalize
from models.ssd import SSD

from chainercv import utils
from chainercv.links import PickableSequentialChain
from chainercv.links import Conv2DActiv

# RGB, (C, 1, 1) format
_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))


class VGG16(PickableSequentialChain):
    """An extended VGG-16 model for SSD300 and SSD512.

    This is an extended VGG-16 model proposed in [#]_.
    The differences from original VGG-16 [#]_ are shown below.

    * :obj:`conv5_1`, :obj:`conv5_2` and :obj:`conv5_3` are changed from \
    :class:`~chainer.links.Convolution2d` to \
    :class:`~chainer.links.DilatedConvolution2d`.
    * :class:`~chainercv.links.model.ssd.Normalize` is \
    inserted after :obj:`conv4_3`.
    * The parameters of max pooling after :obj:`conv5_3` are changed.
    * :obj:`fc6` and :obj:`fc7` are converted to :obj:`conv6` and :obj:`conv7`.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    .. [#] Karen Simonyan, Andrew Zisserman.
       Very Deep Convolutional Networks for Large-Scale Image Recognition.
       ICLR 2015.
    """

    def __init__(self):
        super().__init__()

        with self.init_scope():
            # NOTE: we cannot use Conv2DActiv here since we want to copy
            # the pretrained model, which requires an identical topology
            self.conv1_1 = L.Convolution2D(64, 3, pad=1)
            self.relu1_1 = lambda x: F.relu(x)
            self.conv1_2 = L.Convolution2D(64, 3, pad=1)
            self.relu1_2 = lambda x: F.relu(x)
            self.pool1 = lambda x: F.max_pooling_2d(x, 2)

            self.conv2_1 = L.Convolution2D(128, 3, pad=1)
            self.relu2_1 = lambda x: F.relu(x)
            self.conv2_2 = L.Convolution2D(128, 3, pad=1)
            self.relu2_2 = lambda x: F.relu(x)
            self.pool2 = lambda x: F.max_pooling_2d(x, 2)

            self.conv3_1 = L.Convolution2D(256, 3, pad=1)
            self.relu3_1 = lambda x: F.relu(x)
            self.conv3_2 = L.Convolution2D(256, 3, pad=1)
            self.relu3_2 = lambda x: F.relu(x)
            self.conv3_3 = L.Convolution2D(256, 3, pad=1)
            self.relu3_3 = lambda x: F.relu(x)
            self.pool3 = lambda x: F.max_pooling_2d(x, 2)

            self.conv4_1 = L.Convolution2D(512, 3, pad=1)
            self.relu4_1 = lambda x: F.relu(x)
            self.conv4_2 = L.Convolution2D(512, 3, pad=1)
            self.relu4_2 = lambda x: F.relu(x)
            self.conv4_3 = L.Convolution2D(512, 3, pad=1)
            self.relu4_3 = lambda x: F.relu(x)
            # NOTE: norm4 should be treated as ReLU in adaloss
            # self.norm4 = Normalize(512, initial=initializers.Constant(20))
            self.pool4 = lambda x: F.max_pooling_2d(x, 2)

            self.conv5_1 = L.Convolution2D(512, 3, pad=1)
            self.relu5_1 = lambda x: F.relu(x)
            self.conv5_2 = L.Convolution2D(512, 3, pad=1)
            self.relu5_2 = lambda x: F.relu(x)
            self.conv5_3 = L.Convolution2D(512, 3, pad=1)
            self.relu5_3 = lambda x: F.relu(x)
            self.pool5 = lambda x: F.max_pooling_2d(x, 3, stride=1, pad=1)

            self.conv6 = L.Convolution2D(1024, 3, pad=6, dilate=6)
            self.relu6 = lambda x: F.relu(x)
            self.conv7 = L.Convolution2D(1024, 1)
            self.relu7 = lambda x: F.relu(x)


class VGG16Extractor300(VGG16):
    """A VGG-16 based feature extractor for SSD300.

    This is a feature extractor for :class:`~chainercv.links.model.ssd.SSD300`.
    This extractor is based on :class:`~chainercv.links.model.ssd.VGG16`.
    """

    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self):
        super().__init__()

        init = {
            "initialW": initializers.LeCunUniform(),
            "initial_bias": initializers.Zero(),
        }

        with self.init_scope():
            self.conv8_1 = L.Convolution2D(256, 1, **init)
            self.relu8_1 = lambda x: F.relu(x)
            self.conv8_2 = L.Convolution2D(512, 3, stride=2, pad=1, **init)
            self.relu8_2 = lambda x: F.relu(x)

            self.conv9_1 = L.Convolution2D(128, 1, **init)
            self.relu9_1 = lambda x: F.relu(x)
            self.conv9_2 = L.Convolution2D(256, 3, stride=2, pad=1, **init)
            self.relu9_2 = lambda x: F.relu(x)

            self.conv10_1 = L.Convolution2D(128, 1, **init)
            self.relu10_1 = lambda x: F.relu(x)
            self.conv10_2 = L.Convolution2D(256, 3, **init)
            self.relu10_2 = lambda x: F.relu(x)

            self.conv11_1 = L.Convolution2D(128, 1, **init)
            self.relu11_1 = lambda x: F.relu(x)
            self.conv11_2 = L.Convolution2D(256, 3, **init)
            self.relu11_2 = lambda x: F.relu(x)

        self.pick = (
            "relu4_3",
            "relu7",
            "relu8_2",
            "relu9_2",
            "relu10_2",
            "relu11_2",
        )


class VGG16Extractor512(VGG16):
    """A VGG-16 based feature extractor for SSD512.

    This is a feature extractor for :class:`~chainercv.links.model.ssd.SSD512`.
    This extractor is based on :class:`~chainercv.links.model.ssd.VGG16`.
    """

    insize = 512
    grids = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self):
        super().__init__()

        init = {
            "initialW": initializers.LeCunUniform(),
            "initial_bias": initializers.Zero(),
        }

        with self.init_scope():
            self.conv8_1 = L.Convolution2D(256, 1, **init)
            self.relu8_1 = lambda x: F.relu(x)
            self.conv8_2 = L.Convolution2D(512, 3, stride=2, pad=1, **init)
            self.relu8_2 = lambda x: F.relu(x)

            self.conv9_1 = L.Convolution2D(128, 1, **init)
            self.relu9_1 = lambda x: F.relu(x)
            self.conv9_2 = L.Convolution2D(256, 3, stride=2, pad=1, **init)
            self.relu9_2 = lambda x: F.relu(x)

            self.conv10_1 = L.Convolution2D(128, 1, **init)
            self.relu10_1 = lambda x: F.relu(x)
            self.conv10_2 = L.Convolution2D(256, 3, stride=2, pad=1, **init)
            self.relu10_2 = lambda x: F.relu(x)

            self.conv11_1 = L.Convolution2D(128, 1, **init)
            self.relu11_1 = lambda x: F.relu(x)
            self.conv11_2 = L.Convolution2D(256, 3, stride=2, pad=1, **init)
            self.relu11_2 = lambda x: F.relu(x)

            self.conv12_1 = L.Convolution2D(128, 1, **init)
            self.relu12_1 = lambda x: F.relu(x)
            self.conv12_2 = L.Convolution2D(256, 4, pad=1, **init)
            self.relu12_2 = lambda x: F.relu(x)

        self.pick = (
            "relu4_3",
            "relu7",
            "relu8_2",
            "relu9_2",
            "relu10_2",
            "relu11_2",
            "relu12_2",
        )


class SSD300(SSD):
    """Single Shot Multibox Detector with 300x300 inputs.

    This is a model of Single Shot Multibox Detector [#]_.
    This model uses :class:`~chainercv.links.model.ssd.VGG16Extractor300` as
    its feature extractor.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (string): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the Caffe model provided by \
                `the original implementation \
                <https://github.com/weiliu89/caffe/tree/ssd>`_. \
                The conversion code is `chainercv/examples/ssd/caffe2npz.py`.
            * :obj:`'imagenet'`: Load weights of VGG-16 trained on ImageNet. \
                The weight file is downloaded and cached automatically. \
                This option initializes weights partially and the rests are \
                initialized randomly. In this case, :obj:`n_fg_class` \
                can be set to any number.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _models = {
        "voc0712": {
            "param": {"n_fg_class": 20},
            "url": "https://chainercv-models.preferred.jp/"
            "ssd300_voc0712_converted_2017_06_06.npz",
            "cv2": True,
        },
        "imagenet": {
            "url": "https://chainercv-models.preferred.jp/"
            "ssd_vgg16_imagenet_converted_2017_06_09.npz",
            "cv2": True,
        },
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        param, path = utils.prepare_pretrained_model(
            {"n_fg_class": n_fg_class}, pretrained_model, self._models
        )

        super().__init__(
            extractor=VGG16Extractor300(),
            multibox=Multibox(
                n_class=param["n_fg_class"] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
            ),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=(30, 60, 111, 162, 213, 264, 315),
            mean=_imagenet_mean,
        )

        if path:
            chainer.serializers.load_npz(path, self, strict=False)


class SSD512(SSD):
    """Single Shot Multibox Detector with 512x512 inputs.

    This is a model of Single Shot Multibox Detector [#]_.
    This model uses :class:`~chainercv.links.model.ssd.VGG16Extractor512` as
    its feature extractor.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (string): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the Caffe model provided by \
                `the original implementation \
                <https://github.com/weiliu89/caffe/tree/ssd>`_. \
                The conversion code is `chainercv/examples/ssd/caffe2npz.py`.
            * :obj:`'imagenet'`: Load weights of VGG-16 trained on ImageNet. \
                The weight file is downloaded and cached automatically. \
                This option initializes weights partially and the rests are \
                initialized randomly. In this case, :obj:`n_fg_class` \
                can be set to any number.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _models = {
        "voc0712": {
            "param": {"n_fg_class": 20},
            "url": "https://chainercv-models.preferred.jp/"
            "ssd512_voc0712_converted_2017_06_06.npz",
            "cv2": True,
        },
        "imagenet": {
            "url": "https://chainercv-models.preferred.jp/"
            "ssd_vgg16_imagenet_converted_2017_06_09.npz",
            "cv2": True,
        },
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        param, path = utils.prepare_pretrained_model(
            {"n_fg_class": n_fg_class}, pretrained_model, self._models
        )

        super().__init__(
            extractor=VGG16Extractor512(),
            multibox=Multibox(
                n_class=param["n_fg_class"] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)),
            ),
            steps=(8, 16, 32, 64, 128, 256, 512),
            sizes=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6),
            mean=_imagenet_mean,
        )

        if path:
            chainer.serializers.load_npz(path, self, strict=False)
