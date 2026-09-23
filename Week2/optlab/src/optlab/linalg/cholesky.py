"""[DAY 4] Cholesky factorization — the only linear solver the course needs.

Never invert a matrix to solve a system. Factor once, then two triangular solves:
n³/3 flops instead of n³, and far better numerics.

The failure mode is the feature. `cholesky` succeeds if and only if the matrix is
positive definite, so a failure is information: geometrically the quadratic model has a
direction of non-positive curvature, statistically the Fisher information is singular
and the parameters are not identified from the data.
"""

import numpy as np

from ..errors import NotPositiveDefiniteError
from ..interfaces import ILinearSolver
from ..types import Mat, Vec


def cholesky(A: Mat) -> Mat:
    """The lower-triangular L with A = LLᵀ.

    Raises `NotPositiveDefiniteError` as soon as a pivot is non-positive. Vectorize the
    inner loop over i rather than writing three nested Python loops.
    """
    raise NotImplementedError("[DAY 4] lab 1")


def solve_lower(L: Mat, b: Vec) -> Vec:
    """Forward substitution: solve L y = b for lower-triangular L."""
    raise NotImplementedError("[DAY 4] lab 1")


def solve_upper_from_lower(L: Mat, y: Vec) -> Vec:
    """Back substitution: solve Lᵀ x = y, using L rather than forming its transpose."""
    raise NotImplementedError("[DAY 4] lab 1")


class CholeskySolver(ILinearSolver):
    """Solves A x = b for symmetric positive definite A, via A = LLᵀ.

    Written once on day 4, then injected unchanged into `NewtonDirection`,
    `GaussNewton` and `LevenbergMarquardt`. Three optimizers, one solver, no copies.
    """

    def solve(self, A: Mat, b: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 1")
