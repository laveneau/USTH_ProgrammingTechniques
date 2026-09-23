"""Your Week 1 project, carried into this repository as a gradient oracle.

Replace `dual.py` and `tensor.py` with your own Week 1 code. Nothing in `src/optlab/`
depends on this package — it is used by `tests/` to check the gradients you write here
against an independent, exact computation.
"""

from .dual import Dual, derivative, gradient_forward
from .tensor import Tensor, autodiff_gradient

__all__ = ["Dual", "Tensor", "autodiff_gradient", "derivative", "gradient_forward"]
