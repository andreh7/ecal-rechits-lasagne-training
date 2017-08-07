#!/usr/bin/env python

# from https://github.com/partobs-mdp/Lasagne/blob/1aa7756d6a9f28c41d4de472236fde61d8a54fbc/lasagne/nonlinearities.py#L273
# pull request: https://github.com/Lasagne/Lasagne/pull/843

import theano

class SELU(object):
    """
    Scaled Exponential Linear Unit
    :math:`\\varphi(x)=\\lambda \\left[(x>0) ? x : \\alpha(e^x-1)\\right]`
    The Scaled Exponential Linear Unit (SELU) was introduced in [1]_
    as an activation function that allows the construction of
    self-normalizing neural networks.
    Parameters
    ----------
    scale : float32
        The scale parameter :math:`\\lambda` for scaling all output.
    scale_neg  : float32
        The scale parameter :math:`\\alpha`
        for scaling output for nonpositive argument values.
    Methods
    -------
    __call__(x)
        Apply the SELU function to the activation `x`.
    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import SELU
    >>> selu = SELU(2, 3)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=selu)
    See Also
    --------
    selu: Instance with :math:`\\alpha\\approx1.6733,\\lambda\\approx1.0507`
          as used in [1]_.
    References
    ----------
    .. [1] Guenter Klambauer et al. (2017):
       Self-Normalizing Neural Networks,
       https://arxiv.org/abs/1706.02515
    """
    def __init__(self, scale=1, scale_neg=1):
        self.scale = scale
        self.scale_neg = scale_neg

    def __call__(self, x):
        return self.scale * theano.tensor.switch(
                x > 0.0,
                x,
                self.scale_neg * (theano.tensor.expm1(x)))


selu = SELU(scale=1.0507009873554804934193349852946,
            scale_neg=1.6732632423543772848170429916717)
selu.__doc__ = """selu(x)
    Instance of :class:`SELU` with :math:`\\alpha\\approx 1.6733,
    \\lambda\\approx 1.0507`
    This has a stable and attracting fixed point of :math:`\\mu=0`,
    :math:`\\sigma=1` under the assumptions of the
    original paper on self-normalizing neural networks.
    """

#----------------------------------------------------------------------

# AlphaDropout layer

# https://github.com/Lasagne/Lasagne/pull/855
