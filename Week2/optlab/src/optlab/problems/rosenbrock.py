"""The classic non-convex test function."""

import numpy as np

from ..interfaces import IObjective, ITwiceDifferentiable
from ..types import Mat, Vec


class Rosenbrock(IObjective, ITwiceDifferentiable):
    """f(x) = sum [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2], minimized at x = (1, ..., 1).

    A curved valley: the floor is cheap to reach and then almost flat along a bend, so
    steepest descent crawls. Not convex, so the Hessian is indefinite away from the
    valley - which is what makes it a good stress test for damped Newton on day 4.

    PROVIDED in full. This is a benchmark, not an exercise: it is the fixed obstacle every
    method of the week is timed against, and deriving it would buy you index bookkeeping
    rather than optimization. Read `gradient` anyway - the comment in it describes the
    mistake that makes a hand-written Rosenbrock gradient look right at both ends and be
    wrong in the middle, and `check_gradient` is how you would have caught it.
    """

    def value(self, x: Vec) -> float:
        """f(x), summed over the n-1 coupled terms."""
        gap = x[1:] - x[:-1] ** 2
        return float(np.sum(100.0 * gap**2 + (1.0 - x[:-1]) ** 2))

    def gradient(self, x: Vec) -> Vec:
        """grad f(x). Accumulated, not assigned - see below."""
        # Every term couples x_i with x_{i+1}, so coordinate k is touched TWICE: once as
        # the `i` of its own term, once as the `i+1` of the term before it. Accumulate
        # both contributions rather than assigning, or the ends come out right and the
        # middle silently does not.
        gap = x[1:] - x[:-1] ** 2
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * gap - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * gap
        return g

    def hessian(self, x: Vec) -> Mat:
        """Tridiagonal. Indefinite away from the valley floor - day 4 relies on that."""
        # Tridiagonal for the same reason the gradient accumulated twice: term i couples
        # only x_i with x_{i+1}, so d^2f/dx_idx_j vanishes unless |i - j| <= 1.
        n = x.size
        H = np.zeros((n, n), dtype=np.float64)
        head = np.arange(n - 1)

        # d^2/dx_i^2 of term i: -400(x_{i+1} - x_i^2) + 800x_i^2 + 2, expanded.
        H[head, head] += 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
        # d^2/dx_{i+1}^2 of term i, accumulated on top of the diagonal entry above.
        H[head + 1, head + 1] += 200.0
        # The off-diagonal coupling, written into both triangles so H stays symmetric.
        H[head, head + 1] = -400.0 * x[:-1]
        H[head + 1, head] = -400.0 * x[:-1]
        return H
