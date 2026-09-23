"""The test problem of the week: a quadratic with a condition number you choose."""

import numpy as np

from ..interfaces import IObjective, ITwiceDifferentiable
from ..types import Mat, Vec


class Quadratic(IObjective, ITwiceDifferentiable):
    """f(x) = ½xᵀAx − bᵀx, for symmetric positive definite A.

    Everything about first-order methods is visible here: the gradient descent rate is
    (κ−1)/(κ+1) and momentum improves it to (√κ−1)/(√κ+1), where κ = cond(A). Newton
    reaches the minimum in exactly one step, because the quadratic model is exact.
    """

    def __init__(self, A: Mat, b: Vec) -> None:
        self.A = A
        self.b = b

    def value(self, x: Vec) -> float:
        """[DAY 1]"""
        raise NotImplementedError("[DAY 1] lab 2")

    def gradient(self, x: Vec) -> Vec:
        """[DAY 1] Ax − b."""
        raise NotImplementedError("[DAY 1] lab 2")

    def hessian(self, x: Vec) -> Mat:
        """[DAY 4] A, independent of x."""
        raise NotImplementedError("[DAY 4] lab 2")

    @staticmethod
    def ill_conditioned(n: int, kappa: float) -> "Quadratic":
        """[DAY 1] A diagonal instance with condition number exactly `kappa`.

        Eigenvalues spaced logarithmically between 1 and `kappa`, and b = 0, so the
        minimizer is the origin and the error is just ‖x‖.
        """
        raise NotImplementedError("[DAY 1] lab 2")
