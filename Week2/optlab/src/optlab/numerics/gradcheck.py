"""[DAY 1] Finite-difference derivatives, used to verify every gradient you write.

Your Week 1 autodiff module is the second, independent oracle: finite differences catch
sign and scale errors, autodiff catches them to machine precision. Disagreement between
the two is always worth reading carefully.
"""

from collections.abc import Callable

import numpy as np

from ..types import Mat, Vec


def numerical_gradient(f: Callable[[Vec], float], x: Vec, h: float = 1e-6) -> Vec:
    """Central-difference approximation of ∇f(x).

    Central differences because the error is O(h²) rather than O(h). The default `h` is
    near the optimum of the truncation/rounding trade-off for float64; day 1 asks you to
    plot the error against `h` and find the U-curve yourself.
    """
    raise NotImplementedError("[DAY 1] lab 1")


def numerical_jacobian(f: Callable[[Vec], Vec], x: Vec, h: float = 1e-6) -> Mat:
    """Central-difference approximation of the Jacobian of a vector-valued `f`.

    Shape (m, n) for f: ℝⁿ → ℝᵐ. Reused on day 4 to check a Hessian (the Jacobian of the
    gradient) and on day 5 to check the Jacobian of a residual vector — write it once.
    """
    raise NotImplementedError("[DAY 1] lab 1")


def check_gradient(
    f: Callable[[Vec], float],
    grad: Callable[[Vec], Vec],
    x: Vec,
    tol: float = 1e-5,
) -> float:
    """Compare an analytic gradient against finite differences at `x`.

    Returns the relative error ‖g − g_num‖ / max(1, ‖g_num‖). Raises `AssertionError` if
    it exceeds `tol`, so this can be called directly from a test.
    """
    raise NotImplementedError("[DAY 1] lab 1")


def check_hessian(
    grad: Callable[[Vec], Vec],
    hess: Callable[[Vec], Mat],
    x: Vec,
    tol: float = 1e-5,
) -> float:
    """[DAY 4] Compare an analytic Hessian against the numerical Jacobian of the gradient.

    Note what this reuses: a Hessian *is* the Jacobian of the gradient, so
    `numerical_jacobian` from day 1 does the work. DRY.
    """
    raise NotImplementedError("[DAY 4] lab 2")
