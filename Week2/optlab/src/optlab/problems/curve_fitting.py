"""[DAY 5] Nonlinear least-squares problems for Gauss-Newton and Levenberg-Marquardt.

These implement `ILeastSquaresProblem`, not `IObjective`: GN and LM need r and J
separately. Write the analytic Jacobian, then validate it with `numerical_jacobian`
from day 1 before you let an optimizer anywhere near it.
"""

import numpy as np

from ..interfaces import ILeastSquaresProblem
from ..types import Mat, Vec


class ExpDecay(ILeastSquaresProblem):
    """y = a*exp(-b*x) + c, with parameters w = (a, b, c).

    The day's running example. From a good start Gauss-Newton converges in a handful of
    iterations; from b = 10 the curve has decayed by x ~ 0.5, so b barely changes the model: its
    column of J is tiny, J^T J has a tiny eigenvalue, and GN overshoots while LM copes. Reproducing
    that contrast is the point of the lab.
    """

    def __init__(self, x: Vec, y: Vec) -> None:
        self.x = x
        self.y = y

    def residuals(self, w: Vec) -> Vec:
        """model(x; w) - y."""
        raise NotImplementedError("[DAY 5] lab 1")

    def jacobian(self, w: Vec) -> Mat:
        """dr/d(a, b, c), of shape (n, 3)."""
        raise NotImplementedError("[DAY 5] lab 1")


class GaussianPeak(ILeastSquaresProblem):
    """y = a*exp(-(x - mu)^2 / (2*sigma^2)), with parameters w = (a, mu, sigma)."""

    def __init__(self, x: Vec, y: Vec) -> None:
        self.x = x
        self.y = y

    def residuals(self, w: Vec) -> Vec:
        raise NotImplementedError("[DAY 5] lab 1")

    def jacobian(self, w: Vec) -> Mat:
        raise NotImplementedError("[DAY 5] lab 1")


class Sinusoid(ILeastSquaresProblem):
    """y = a*sin(omega*x + phi), with parameters w = (a, omega, phi).

    The counterexample: start with omega far from the truth and LM converges neatly to a
    *local* minimum. "Converged" does not mean "best" - neither GN nor LM is a global
    method, and nothing in this course is.
    """

    def __init__(self, x: Vec, y: Vec) -> None:
        self.x = x
        self.y = y

    def residuals(self, w: Vec) -> Vec:
        raise NotImplementedError("[DAY 5] lab 1")

    def jacobian(self, w: Vec) -> Mat:
        raise NotImplementedError("[DAY 5] lab 1")
