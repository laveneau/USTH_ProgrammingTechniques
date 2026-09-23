"""[DAY 5] Nonlinear least-squares problems for Gauss-Newton and Levenberg-Marquardt.

These implement `LeastSquaresProblem`, not `Objective`: GN and LM need r and J
separately. Write the analytic Jacobian, then validate it with `numerical_jacobian`
from day 1 before you let an optimizer anywhere near it.
"""

import numpy as np

from ..interfaces import LeastSquaresProblem
from ..types import Mat, Vec


class ExpDecay(LeastSquaresProblem):
    """y = a·exp(−b·t) + c, with parameters x = (a, b, c).

    The day's running example. From a good start Gauss-Newton converges in a handful of
    iterations; from b = 10 the model is nearly constant past t ≈ 0.5, so a and b are
    poorly determined, JᵀJ is near-singular, and GN drifts while LM copes. Reproducing
    that contrast is the point of the lab.
    """

    def __init__(self, t: Vec, y: Vec) -> None:
        self.t = t
        self.y = y

    def residuals(self, x: Vec) -> Vec:
        """model(t; x) − y."""
        raise NotImplementedError("[DAY 5] lab 1")

    def jacobian(self, x: Vec) -> Mat:
        """∂r/∂(a, b, c), of shape (m, 3)."""
        raise NotImplementedError("[DAY 5] lab 1")


class GaussianPeak(LeastSquaresProblem):
    """y = a·exp(−(t − mu)² / (2σ²)), with parameters x = (a, mu, sigma)."""

    def __init__(self, t: Vec, y: Vec) -> None:
        self.t = t
        self.y = y

    def residuals(self, x: Vec) -> Vec:
        raise NotImplementedError("[DAY 5] lab 1")

    def jacobian(self, x: Vec) -> Mat:
        raise NotImplementedError("[DAY 5] lab 1")


class Sinusoid(LeastSquaresProblem):
    """y = a·sin(omega·t + phi), with parameters x = (a, omega, phi).

    The counterexample: start with omega far from the truth and LM converges neatly to a
    *local* minimum. "Converged" does not mean "best" — neither GN nor LM is a global
    method, and nothing in this course is.
    """

    def __init__(self, t: Vec, y: Vec) -> None:
        self.t = t
        self.y = y

    def residuals(self, x: Vec) -> Vec:
        raise NotImplementedError("[DAY 5] lab 1")

    def jacobian(self, x: Vec) -> Mat:
        raise NotImplementedError("[DAY 5] lab 1")
