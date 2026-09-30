"""[DAY 5] Nonlinear least squares: exploiting the structure of 1/2||r(x)||^2.

The exact Hessian is J^T J + sum r_i grad^2 r_i. Gauss-Newton drops the second term, which costs
nothing to justify when the residuals are small or the model is nearly linear, and costs
convergence when they are not.

Both classes take an `ILinearSolver` and both reuse the day-4 Cholesky unchanged.
"""

import numpy as np

from ..interfaces import ILeastSquaresProblem, ILinearSolver, IObserver, IOptimizer
from ..results import OptimizeResult
from ..types import Vec


class GaussNewton(IOptimizer):
    """Solve (J^T J)*delta = -J^T r, then x <- x + delta.

    Cheap - only first derivatives - and nearly quadratic when residuals are small. But
    J^T J can be singular, and then it diverges. Day 5 asks you to make it diverge on
    purpose from the hard start and to document that failure rather than patch it: it is
    the motivation for the next class.
    """

    def __init__(
        self,
        linear_solver: ILinearSolver,
        tol: float = 1e-8,
        max_iter: int = 100,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.linear_solver = linear_solver
        self.tol = tol
        self.max_iter = max_iter
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `ILeastSquaresProblem`."""
        raise NotImplementedError("[DAY 5] lab 2")


class LevenbergMarquardt(IOptimizer):
    """Solve (J^T J + lam*I)*delta = -J^T r, adapting lam by the gain ratio.

    lam -> 0 recovers Gauss-Newton; lam -> inf gives a small gradient step. Since J^T J + lam*I is
    positive definite for any lam > 0, Cholesky never fails here - a direct payoff of
    day 4, and the reason LM is the workhorse of curve fitting.

    lam is the dual of a trust-region radius: larger lam means a smaller trusted step. The
    gain ratio

        rho = (actual reduction) / (predicted reduction),   predicted = 1/2*delta^T(lam*delta - g)

    drives it. rho small or negative: reject the step and increase lam (shrink the region).
    rho large: accept and decrease lam. This is day 4's damped Newton with a principled tau.

    Line search fixes a direction and searches the length; a trust region fixes a radius
    and searches direction and length together inside the ball. LM is the second kind.
    """

    def __init__(
        self,
        linear_solver: ILinearSolver,
        lambda0: float = 1e-3,
        lambda_up: float = 10.0,
        lambda_down: float = 0.1,
        tol: float = 1e-8,
        max_iter: int = 200,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.linear_solver = linear_solver
        self.lambda0 = lambda0
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.tol = tol
        self.max_iter = max_iter
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `ILeastSquaresProblem`."""
        raise NotImplementedError("[DAY 5] lab 3")
