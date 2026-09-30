"""The classic non-convex test function."""

import numpy as np

from ..interfaces import IObjective, ITwiceDifferentiable
from ..types import Mat, Vec


class Rosenbrock(IObjective, ITwiceDifferentiable):
    """f(x) = sum [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2], minimized at x = (1, ..., 1).

    A curved valley: the floor is cheap to reach and then almost flat along a bend, so
    steepest descent crawls. Not convex, so the Hessian is indefinite away from the
    valley - which is what makes it a good stress test for damped Newton on day 4.
    """

    def value(self, x: Vec) -> float:
        """[DAY 1]"""
        raise NotImplementedError("[DAY 1] lab 2")

    def gradient(self, x: Vec) -> Vec:
        """[DAY 1] Check it against finite differences before you trust it."""
        raise NotImplementedError("[DAY 1] lab 2")

    def hessian(self, x: Vec) -> Mat:
        """[DAY 4] Tridiagonal. Indefinite away from the valley floor."""
        raise NotImplementedError("[DAY 4] lab 2")
