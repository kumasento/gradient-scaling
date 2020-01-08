from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.transforms import resize
from chainercv import utils

from ada_loss.chainer_impl.ada_loss import AdaLossChainer
from ada_loss.chainer_impl.functions.ada_loss_cast import ada_loss_cast

class SegNetBasic(chainer.Chain):

    """SegNet Basic for semantic segmentation.

    This is a SegNet [#]_ model for semantic segmenation. This is based on
    SegNetBasic model that is found here_.

    When you specify the path of a pretrained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`camvid`: Loads weights trained with the train split of \
        CamVid dataset.

    .. [#] Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A \
    Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." \
    PAMI, 2017

    .. _here: http://github.com/alexgkendall/SegNet-Tutorial

    Args:
        n_class (int): The number of classes. If :obj:`None`, it can
            be infered if :obj:`pretrained_model` is given.
        pretrained_model (string): The destination of the pretrained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        initialW (callable): Initializer for convolution layers.

    """

    _models = {
        'camvid': {
            'param': {'n_class': 11},
            'url': 'https://chainercv-models.preferred.jp/'
            'segnet_camvid_trained_2018_12_05.npz'
        }
    }

    def __init__(self, n_class=None, pretrained_model=None, initialW=None, dtype='float32'):
        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class}, pretrained_model, self._models)
        self.n_class = param['n_class']
        self.dtype = dtype

        if initialW is None:
            initialW = chainer.initializers.HeNormal()

        super(SegNetBasic, self).__init__()
        with self.init_scope():
            self.lrn = lambda x: F.local_response_normalization(x, 5, 1, 1e-4 / 5., 0.75)

            self.conv1 = L.Convolution2D(
                None, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv1_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv1_relu = lambda x: F.relu(x)
            self.conv1_pool = lambda x: F.max_pooling_2d(x, 2, 2, return_indices=True)

            self.conv2 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv2_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv2_relu = lambda x: F.relu(x)
            self.conv2_pool = lambda x: F.max_pooling_2d(x, 2, 2, return_indices=True)

            self.conv3 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv3_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv3_relu = lambda x: F.relu(x)
            self.conv3_pool = lambda x: F.max_pooling_2d(x, 2, 2, return_indices=True)

            self.conv4 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv4_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv4_relu = lambda x: F.relu(x)
            self.conv4_pool = lambda x: F.max_pooling_2d(x, 2, 2, return_indices=True)

            self.upsampling4 = lambda x, indices: self._upsampling_2d(x, indices)
            self.conv_decode4 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv_decode4_bn = L.BatchNormalization(64, initial_beta=0.001)

            self.upsampling3 = lambda x, indices: self._upsampling_2d(x, indices)
            self.conv_decode3 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv_decode3_bn = L.BatchNormalization(64, initial_beta=0.001)
            
            self.upsampling2 = lambda x, indices: self._upsampling_2d(x, indices)
            self.conv_decode2 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv_decode2_bn = L.BatchNormalization(64, initial_beta=0.001)

            self.upsampling1 = lambda x, indices: self._upsampling_2d(x, indices)
            self.conv_decode1 = L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=initialW)
            self.conv_decode1_bn = L.BatchNormalization(64, initial_beta=0.001)
            self.conv_classifier = L.Convolution2D(
                64, self.n_class, 1, 1, 0, initialW=initialW)

        self.type_cast_ada_loss = None

        if path:
            chainer.serializers.load_npz(path, self)

    def _upsampling_2d(self, x, indices):
        if x.shape != indices.shape:
            min_h = min(x.shape[2], indices.shape[2])
            min_w = min(x.shape[3], indices.shape[3])
            x = x[:, :, :min_h, :min_w]
            indices = indices[:, :, :min_h, :min_w]
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(x, indices.array, ksize=2, stride=2, outsize=outsize)

    def forward(self, x):
        """Compute an image-wise score from a batch of images

        Args:
            x (chainer.Variable): A variable with 4D image array.

        Returns:
            chainer.Variable:
            An image-wise score. Its channel size is :obj:`self.n_class`.

        """
        # h = F.local_response_normalization(x, 5, 1, 1e-4 / 5., 0.75)
        # h, indices1 = F.max_pooling_2d(
        #     F.relu(self.conv1_bn(self.conv1(h))), 2, 2, return_indices=True)
        # h, indices2 = F.max_pooling_2d(
        #     F.relu(self.conv2_bn(self.conv2(h))), 2, 2, return_indices=True)
        # h, indices3 = F.max_pooling_2d(
        #     F.relu(self.conv3_bn(self.conv3(h))), 2, 2, return_indices=True)
        # h, indices4 = F.max_pooling_2d(
        #     F.relu(self.conv4_bn(self.conv4(h))), 2, 2, return_indices=True)
        # h = self._upsampling_2d(h, indices4)
        # h = self.conv_decode4_bn(self.conv_decode4(h))
        # h = self._upsampling_2d(h, indices3)
        # h = self.conv_decode3_bn(self.conv_decode3(h))
        # h = self._upsampling_2d(h, indices2)
        # h = self.conv_decode2_bn(self.conv_decode2(h))
        # h = self._upsampling_2d(h, indices1)
        # h = self.conv_decode1_bn(self.conv_decode1(h))

        h = self.lrn(x) 

        h, indices1 = self.conv1_pool(self.conv1_relu(self.conv1_bn(self.conv1(h))))
        h, indices2 = self.conv2_pool(self.conv2_relu(self.conv2_bn(self.conv2(h))))
        h, indices3 = self.conv3_pool(self.conv3_relu(self.conv3_bn(self.conv3(h))))
        h, indices4 = self.conv4_pool(self.conv4_relu(self.conv4_bn(self.conv4(h))))

        h = self.upsampling4(h, indices4)
        h = self.conv_decode4_bn(self.conv_decode4(h))
        h = self.upsampling3(h, indices3)
        h = self.conv_decode3_bn(self.conv_decode3(h))
        h = self.upsampling2(h, indices2)
        h = self.conv_decode2_bn(self.conv_decode2(h))
        h = self.upsampling1(h, indices1)
        h = self.conv_decode1_bn(self.conv_decode1(h))

        h = self.conv_classifier(h)

        # TODO: refactorize this. Instead of hardcoding, use AdaLossScaled
        if self.dtype != 'float32':
            if self.type_cast_ada_loss is None:
                self.type_cast_ada_loss = AdaLossChainer(**self.conv1_bn.ada_loss_cfg)
            h = ada_loss_cast(h, 'float32', self.type_cast_ada_loss)

        return h 

    def predict(self, imgs):
        """Conduct semantic segmentations from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.

        Returns:
            list of numpy.ndarray:

            List of integer labels predicted from each image in the input \
            list.

        """
        labels = []
        for img in imgs:
            C, H, W = img.shape
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
              
                x = F.cast(self.xp.asarray(img[np.newaxis]), self.dtype)
                score = self.forward(x)[0].array.astype(np.float32)

            score = chainer.backends.cuda.to_cpu(score)
            if score.shape != (C, H, W):
                dtype = score.dtype
                score = resize(score, (H, W)).astype(dtype)

            label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels
