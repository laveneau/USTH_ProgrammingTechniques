"""[DAY 5] Nonlinear least squares: exploiting the structure of ½‖r(x)‖².

The exact Hessian is JᵀJ + Σ rᵢ∇²rᵢ. Gauss-Newton drops the second term, which costs
nothing to justify when the residuals are small or the model is nearly linear, and costs
convergence when they are not.

Both classes take a `LinearSolver` and both reuse the day-4 Cholesky unchanged.
"""

import numpy as np

from ..interfaces import LeastSquaresProblem, LinearSolver, Observer, Optimizer
from ..results import OptimizeResult
from ..types import Vec


class GaussNewton(Optimizer):
    """Solve (JᵀJ)δ = −Jᵀr, then x ← x + δ.

    Cheap — only first derivatives — and nearly quadratic when residuals are small. But
    JᵀJ can be singular, and then it diverges. Day 5 asks you to make it diverge on
    purpose from the hard start and to document that failure rather than patch it: it is
    the motivation for the next class.
    """

    def __init__(
        self,
        linear_solver: LinearSolver,
        tol: float = 1e-8,
        max_iter: int = 100,
        observers: list[Observer] | None = None,
    ) -> None:
        self.linear_solver = linear_solver
        self.tol = tol
        self.max_iter = max_iter
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires a `LeastSquaresProblem`."""
        raise NotImplementedError("[DAY 5] lab 2")


class LevenbergMarquardt(Optimizer):
    """Solve (JᵀJ + λI)δ = −Jᵀr, adapting λ by the gain ratio.

    λ → 0 recovers Gauss-Newton; λ → ∞ gives a small gradient step. Since JᵀJ + λI is
    positive definite for any λ > 0, Cholesky never fails here — a direct payoff of
    day 4, and the reason LM is the workhorse of curve fitting.

    λ is the dual of a trust-region radius: larger λ means a smaller trusted step. The
    gain ratio

        ρ = (actual reduction) / (predicted reduction),   predicted = ½δᵀ(λδ − g)

    drives it. ρ small or negative: reject the step and increase λ (shrink the region).
    ρ large: accept and decrease λ. This is day 4's damped Newton with a principled τ.

    Line search fixes a direction and searches the length; a trust region fixes a radius
    and searches direction and length together inside the ball. LM is the second kind.
    """

    def __init__(
        self,
        linear_solver: LinearSolver,
        lambda0: float = 1e-3,
        lambda_up: float = 10.0,
        lambda_down: float = 0.1,
        tol: float = 1e-8,
        max_iter: int = 200,
        observers: list[Observer] | None = None,
    ) -> None:
        self.linear_solver = linear_solver
        self.lambda0 = lambda0
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.tol = tol
        self.max_iter = max_iter
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires a `LeastSquaresProblem`."""
        raise NotImplementedError("[DAY 5] lab 3")
