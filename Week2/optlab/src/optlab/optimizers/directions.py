"""Search directions. One class per rule; the loop never changes."""

import numpy as np

from ..errors import NotPositiveDefiniteError
from ..interfaces import IDirectionRule, ILinearSolver, IObjective
from ..types import Vec


class SteepestDescent(IDirectionRule):
    """[DAY 2] d = −g. The obvious choice, and the slow one on an ill-conditioned problem.

    On a quadratic with condition number κ the error contracts by (κ−1)/(κ+1) per step:
    about 690 iterations for ½(x₁² + 100x₂²) where momentum needs 70 and Newton needs 1.
    """

    def direction(self, objective: IObjective, x: Vec, g: Vec) -> Vec:
        raise NotImplementedError("[DAY 2] lab 1")


class HeavyBall(IDirectionRule):
    """[DAY 2] d = −g + beta·(previous step). Polyak momentum.

    Stateful: it remembers the last direction, which is why an `IDirectionRule` is an
    object and not a function. Improves the rate to (√κ−1)/(√κ+1) — a square root, which
    on κ = 10⁴ is the difference between 10⁴ and 10² iterations.
    """

    def __init__(self, beta: float = 0.9) -> None:
        self.beta = beta
        self._previous: Vec | None = None

    def direction(self, objective: IObjective, x: Vec, g: Vec) -> Vec:
        raise NotImplementedError("[DAY 2] lab 3")


class NewtonDirection(IDirectionRule):
    """[DAY 4] Solve H d = −g for the Newton direction.

    The objective must be `ITwiceDifferentiable`. Receives its `ILinearSolver` by
    constructor rather than calling Cholesky directly — so the same class becomes
    Newton-CG the day someone writes a conjugate-gradient solver, with no edit here.

    Propagate `NotPositiveDefiniteError` rather than silently falling back to the
    gradient: an indefinite Hessian is a fact about the problem the caller should see.
    Day 4 lab 3 adds the damping H + τI that deals with it.
    """

    def __init__(self, linear_solver: ILinearSolver) -> None:
        self.linear_solver = linear_solver

    def direction(self, objective: IObjective, x: Vec, g: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 2")


class ModifiedNewton(IDirectionRule):
    """[DAY 4] Newton with the damping H + τI that makes an indefinite Hessian usable.

    `NewtonDirection` propagates `NotPositiveDefiniteError` because an indefinite Hessian
    is a fact the caller should see. This class is the repair: catch it, add τI, and
    double τ until Cholesky succeeds. Since H + τI → τI as τ grows, and the solution of
    τI·d = −g is −g/τ, large τ degrades gracefully to a small gradient step — you are
    never worse off than steepest descent.

    Start each call from `tau0` rather than from the τ that worked last time: once you are
    near a minimum the Hessian is positive definite again, and you want α = 1 Newton steps
    back, not a permanently damped method.

    Raise `NotPositiveDefiniteError` if τ exceeds `tau_max`; that means something is wrong
    with the Hessian itself (a NaN, or a sign error in your second derivative), and
    silently returning −g would hide it.

    Tomorrow this same H + τI reappears as Levenberg–Marquardt, where τ is called λ and is
    chosen by a trust-region rule rather than by doubling.
    """

    def __init__(
        self,
        linear_solver: ILinearSolver,
        tau0: float = 1e-3,
        tau_max: float = 1e12,
    ) -> None:
        self.linear_solver = linear_solver
        self.tau0 = tau0
        self.tau_max = tau_max

    def direction(self, objective: IObjective, x: Vec, g: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 3")
