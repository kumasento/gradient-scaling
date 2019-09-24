""" Base class that defines the transformation. """


class AdaLossTransform(object):
    """ The base class """
    cls = None

    def __call__(self, link, cfg):
        """ Entry """
        assert isinstance(link, self.cls)
        new_link = self.create(link, cfg)
        self.copyparams(link, new_link)
        return new_link

    def create(self, link, cfg):
        """ Create the new link """
        raise NotImplementedError('create() should be implemented')

    def copyparams(self, src, dst):
        """ Copy parameter from the old link to the new link """
        dst.copyparams(src)