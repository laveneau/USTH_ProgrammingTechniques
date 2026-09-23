"""[DAY 2] How far to move along a direction."""

from .errors import LineSearchFailed
from .interfaces import LineSearch, Objective
from .types import Vec


class FixedStep(LineSearch):
    """A constant step length, ignoring the objective entirely.

    Safe when alpha <= 1/L for an L-smooth f; diverges above 2/L, which day 2 asks you
    to trigger on purpose and explain.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def step(self, objective: Objective, x: Vec, g: Vec, d: Vec) -> float:
        raise NotImplementedError("[DAY 2] lab 1")


class Armijo(LineSearch):
    """Backtracking until the sufficient-decrease condition holds:

        f(x + αd) <= f(x) + c1·α·gᵀd

    Start at `alpha0`, multiply by `rho` on rejection, give up after `max_backtracks`
    and raise `LineSearchFailed`. The optimizer catches that and reports
    `converged=False`; it must not crash.

    Injecting this in place of `FixedStep` must require no edit to `DescentOptimizer`.
    If it does, the loop is doing something that belongs here.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        c1: float = 1e-4,
        rho: float = 0.5,
        max_backtracks: int = 50,
    ) -> None:
        self.alpha0 = alpha0
        self.c1 = c1
        self.rho = rho
        self.max_backtracks = max_backtracks

    def step(self, objective: Objective, x: Vec, g: Vec, d: Vec) -> float:
        raise NotImplementedError("[DAY 2] lab 2")
