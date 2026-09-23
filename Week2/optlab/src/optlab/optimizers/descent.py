"""[DAY 2] The one deterministic descent loop, written once and never again.

Gradient descent, momentum, Newton and ridge are all this loop with different objects
plugged in. If you find yourself editing it on day 4 to make Newton work, something
belongs in an `IDirectionRule` or an `ILineSearch` instead.
"""

import numpy as np

from ..errors import LineSearchFailed
from ..interfaces import (
    IDirectionRule,
    ILineSearch,
    IObjective,
    IObserver,
    IOptimizer,
    IStoppingCriterion,
)
from ..linalg import CholeskySolver
from ..linesearch import Armijo, FixedStep
from ..results import OptimizeResult, StepEvent
from ..stopping import AnyOf, GradientNormBelow, MaxIterations
from ..types import Vec
from .directions import HeavyBall, NewtonDirection, SteepestDescent


class DescentOptimizer(IOptimizer):
    """x ← x + α·d, where `direction` chooses d and `line_search` chooses α.

    Every collaborator arrives through the constructor. The loop itself only:
    evaluates, asks for a direction, asks for a step, moves, builds a `StepEvent`,
    notifies the observers, and asks the criterion whether to stop.

    Catch `LineSearchFailed` and return `converged=False` with the reason in `message`.
    """

    def __init__(
        self,
        direction: IDirectionRule,
        line_search: ILineSearch,
        stop: IStoppingCriterion,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.direction = direction
        self.line_search = line_search
        self.stop = stop
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `IObjective`."""
        raise NotImplementedError("[DAY 2] lab 1")


def gradient_descent(tol: float = 1e-6, max_iter: int = 1000) -> DescentOptimizer:
    """[DAY 2] Steepest descent with an Armijo line search."""
    raise NotImplementedError("[DAY 2] lab 2")


def momentum(beta: float = 0.9, alpha: float = 0.01, max_iter: int = 1000) -> DescentOptimizer:
    """[DAY 2] Heavy ball with a fixed step."""
    raise NotImplementedError("[DAY 2] lab 3")


def newton(tol: float = 1e-8, max_iter: int = 100) -> DescentOptimizer:
    """[DAY 4] Newton's direction, Cholesky solver, Armijo starting at α = 1.

    Note what this function is: four existing objects composed. No new loop, no new
    optimizer class. If it reads as anything more than wiring, the wiring is wrong.
    """
    raise NotImplementedError("[DAY 4] lab 2")
