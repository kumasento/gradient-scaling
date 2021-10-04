""" A link that wrap adaptive loss scaling around an existing link. """

# from inspect import signature
from types import MethodType
from collections import OrderedDict

import chainer
import chainer.links as L
from chainercv.links import PickableSequentialChain

from ada_loss.chainer_impl.utils import scale_grad, set_random_seed
from ada_loss.chainer_impl.links import identity_loss_scaling
from ada_loss.chainer_impl.functions.loss_scaling import loss_scaling
from ada_loss.chainer_impl.functions.ada_loss_branch import AdaLossBranch
from ada_loss.chainer_impl import ada_loss_transforms

# NOTE: default transforms to be performed
default_transforms = [
    v()
    for k, v in ada_loss_transforms.__dict__.items()
    if k in ada_loss_transforms.__all__ and callable(v)
]


def ada_loss_forward_for_pickable(self, x, verbose=False):
    """ """
    assert isinstance(self, PickableSequentialChain)

    if self._pick is None:
        pick = (self.layer_names[-1],)
    else:
        pick = self._pick
    last_index = max(self.layer_names.index(name) for name in pick)

    layers = {}
    h = x
    for i, name in enumerate(self.layer_names[: last_index + 1]):
        h = self[name](h)
        if name in pick:
            if i != last_index:
                # print('==> Branching picked layer {}'.format(name))
                h1, h2 = AdaLossBranch().apply((h,))
                layers[name] = h2
                h = h1
            else:
                layers[name] = h

    if self._return_tuple:
        layers = tuple(layers[name] for name in pick)
    else:
        layers = list(layers.values())[0]
    return layers


class AdaLossScaled(chainer.Chain):
    """ Given an input link, wrap it to support adaptive loss scaling. """

    def __init__(
        self,
        link,
        init_scale=1.0,
        cfg=None,
        transforms=default_transforms,
        transform_functions=True,
        seed=None,
        verbose=False,
    ):
        """ CTOR. """
        super().__init__()

        assert isinstance(link, chainer.Link), "link should be a chainer.Link"

        self.verbose = verbose
        self.transform_functions = transform_functions
        # scale up the very first input gradient
        self.init_scale = init_scale
        # wrapped link
        if transforms is None:
            transforms = []
        if cfg is None:
            cfg = {}
        if seed is not None:
            set_random_seed(seed)
        self.cfg = cfg
        print(cfg)
        self.setup(link, cfg=cfg, transforms=transforms)

        with self.init_scope():
            self.link = link

    def predict(self, x):
        return self.link.predict(x)

    def forward(self, x):
        """ Forward computation """
        return loss_scaling(self.link(x), self.init_scale)

    def setup(self, link, cfg=None, transforms=None):
        """ Wrap the link that will be computed. """
        # decide the new mapping
        new_dict = OrderedDict()  # NOTE: keep the sequence

        # NOTE: please choose Python version >= 3.6 or the order will be distorted.
        # https://www.python.org/dev/peps/pep-0520/
        for attr, value in link.__dict__.items():
            if callable(value):  # function and Link are all belongs to this
                if self.verbose:
                    print('==> Attribute "{}" is callable'.format(attr))

                if isinstance(value, chainer.Link):
                    # recursively setup all links
                    new_dict[attr] = self.setup_link(
                        value, cfg=cfg, transforms=transforms
                    )
                else:  # stop here
                    if self.verbose:
                        print("==> Wrapping function {} ...".format(attr))
                    if self.transform_functions:
                        new_dict[attr] = self.wrap_func(value)

        # perform the update
        # NOTE: if there is any picked layer, cache them
        pick = None
        if isinstance(link, PickableSequentialChain):
            pick = link.pick
            link.pick = None  # unset

        # NOTE: here we assume all layers in link are updated in their original sequence
        with link.init_scope():
            for attr, value in new_dict.items():
                if self.verbose:
                    print(
                        '==> Replacing attribute "{}" from {} to {} ...'.format(
                            attr, getattr(link, attr), value
                        )
                    )
                delattr(link, attr)
                setattr(link, attr, value)

        if pick is not None:
            link.pick = pick

        self.update_link_methods(link)
        return link

    def setup_link(self, link, cfg=None, transforms=None):
        """ Try to transform a link """
        if transforms is None:
            transforms = {}
        for trans in transforms:
            if not isinstance(link, trans.cls):
                continue
            if self.verbose:
                print(
                    "==> Transforming link of type {} with cfg {} ...".format(
                        type(link), cfg
                    )
                )
            return trans(link, cfg=cfg)

        # go deeper into the link
        return self.setup(link, cfg=cfg, transforms=transforms)

    def wrap_func(self, func):
        """ Wrap the function with loss scaling.
            Returns a link.
        """
        # if len(signature(func).parameters) != 1:
        #     raise ValueError(
        #         'Cannot support function with more than one parameters: sig {}'
        #         .format(signature(func)))
        return identity_loss_scaling.IdentityLossScalingWrapper(func)

    def update_link_methods(self, link):
        """ Update the default behaviour of the link """
        if isinstance(link, PickableSequentialChain):
            return self.update_pickable_link_methods(link)

        return link

    def update_pickable_link_methods(self, link):
        """ """
        assert isinstance(link, PickableSequentialChain)

        if link.pick is None:  # No need to update
            return link
        if self.verbose:
            print("==> Updating the forward method of link {} ...".format(link))

        link.forward = MethodType(ada_loss_forward_for_pickable, link)
        return link
