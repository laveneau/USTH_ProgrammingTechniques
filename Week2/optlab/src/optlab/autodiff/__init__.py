"""PROVIDED - automatic differentiation over numpy arrays, used as a gradient oracle.

This is Week 1's project carried forward and generalized from numbers to arrays. Nothing
in `src/optlab/` depends on it: it exists so that `tests/` can check every gradient you
write by hand against an independent computation that is exact to machine precision,
alongside the finite differences of `numerics/gradcheck.py`.

`autodiff_gradient` (reverse mode) is the one used from day 1 - one sweep for a whole
gradient, whatever the number of parameters. `jacobian_forward` (forward mode) is the one
day 5 wants, where there are three parameters and many residuals.

Your own Week 1 module works on single numbers, and day 1's labwork asks you to compare
the two. `tensor.py` explains the three things that had to change.
"""

from .dual import Dual, derivative, gradient_forward, jacobian_forward
from .functions import cos, exp, log, sin
from .tensor import Tensor, autodiff_gradient, unbroadcast

__all__ = [
    "Dual",
    "Tensor",
    "autodiff_gradient",
    "cos",
    "derivative",
    "exp",
    "gradient_forward",
    "jacobian_forward",
    "log",
    "sin",
    "unbroadcast",
]
