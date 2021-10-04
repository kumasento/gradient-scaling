""" Passing element-wise multiplication results through threshold
    before adding them altogether. """

from chainer import utils
from chainer.links.connection.linear import Linear

from chainerlp.functions import thresholded_linear


class ThresholdedLinear(Linear):
    """ Inherited from the Linear link in chainer. """

    def __init__(
        self,
        in_size,
        out_size=None,
        nobias=False,
        initialW=None,
        initial_bias=None,
        threshold=6e-8,
    ):
        """ """
        super(ThresholdedLinear, self).__init__(
            in_size,
            out_size=out_size,
            nobias=nobias,
            initialW=initialW,
            initial_bias=initial_bias,
        )
        self.threshold = threshold

    def forward(self, x, n_batch_axes=1):
        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[1:])
            self._initialize_params(in_size)

        return thresholded_linear(
            x, self.W, self.b, n_batch_axes=n_batch_axes, threshold=self.threshold
        )

    @property
    def printable_specs(self):
        specs = [
            ("in_size", self.in_size),
            ("out_size", self.out_size),
            ("nobias", self.b is None),
            ("threshold", self.threshold),
        ]
        for spec in specs:
            yield spec
