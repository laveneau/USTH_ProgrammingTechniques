"""[DAY 1] Finite-difference derivatives, used to verify every gradient you write.

You write exactly one function here, `numerical_gradient`, because the central difference
and its step size are the content of Lecture 1 Sec. 8. The other three are PROVIDED and
meant to be read: `numerical_jacobian` is the same sweep with a column instead of a
number, and the two `check_*` wrappers are the same five lines twice - take a norm, divide
by a scale, assert.

`optlab.autodiff` is the second, independent oracle: finite differences catch sign and
scale errors and stop at about six correct digits, autodiff has no step size and is exact
to machine precision. Disagreement between the two is always worth reading carefully.
"""

from collections.abc import Callable

import numpy as np

from ..types import Mat, Vec


def numerical_gradient(f: Callable[[Vec], float], x: Vec, h: float = 1e-6) -> Vec:
    """Central-difference approximation of grad f(x).

    Central differences because the error is O(h^2) rather than O(h). The default `h` is
    near the optimum of the truncation/rounding trade-off for float64; day 1 asks you to
    plot the error against `h` and find the U-curve yourself.
    """
    raise NotImplementedError("[DAY 1] lab 1")


def numerical_jacobian(f: Callable[[Vec], Vec], x: Vec, h: float = 1e-6) -> Mat:
    """Central-difference approximation of the Jacobian of a vector-valued `f`.

    Shape (m, n) for f: R^n -> R^m. Reused on day 4 to check a Hessian (the Jacobian of the
    gradient) and on day 5 to check the Jacobian of a residual vector - write it once.
    """
    m = f(x).size
    jac = np.zeros((m, x.size), dtype=np.float64)
    step = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        # Same sweep as above, except each difference is a whole column of the Jacobian.
        step[i] = h
        jac[:, i] = (f(x + step) - f(x - step)) / (2.0 * h)
        step[i] = 0.0
    return jac


def check_gradient(
    f: Callable[[Vec], float],
    grad: Callable[[Vec], Vec],
    x: Vec,
    tol: float = 1e-5,
) -> float:
    """Compare an analytic gradient against finite differences at `x`.

    Returns the relative error ||g - g_num|| / max(1, ||g_num||). Raises `AssertionError` if
    it exceeds `tol`, so this can be called directly from a test.
    """
    analytic = grad(x)
    numeric = numerical_gradient(f, x)
    # max(1, ||g_num||) rather than ||g_num||: near a minimum the gradient goes to zero, and
    # dividing by it would turn a harmless absolute error into a huge relative one.
    scale = max(1.0, float(np.linalg.norm(numeric)))
    error = float(np.linalg.norm(analytic - numeric)) / scale
    assert error <= tol, (
        f"gradient check failed: relative error {error:.3e} > tol {tol:.3e}\n"
        f"  analytic: {analytic}\n"
        f"  numeric : {numeric}"
    )
    return error


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
    # Not a second finite-difference routine: the day-1 sweep, handed the gradient
    # instead of the value. Every column of the result is d(grad f)/dx_i, which is a column
    # of the Hessian.
    analytic = hess(x)
    numeric = numerical_jacobian(grad, x)
    scale = max(1.0, float(np.linalg.norm(numeric)))
    error = float(np.linalg.norm(analytic - numeric)) / scale
    assert error <= tol, (
        f"hessian check failed: relative error {error:.3e} > tol {tol:.3e}\n"
        f"  analytic:\n{analytic}\n"
        f"  numeric :\n{numeric}"
    )
    return error
