import chainer
import six


class LoadOnlyUpdator(chainer.training.StandardUpdater):
    """ This updator is designed for only serializing model. """

    def serialize(self, serializer):
        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.serialize(serializer["model:" + name])
