"""Every abstract base class of optlab. PROVIDED — do not edit.

These are the contracts of the week. You write implementations that inherit them; you
never change an interface. An implementation may inherit several — for example
`GLMLoss(Objective, TwiceDifferentiable, BatchObjective)` — which is the multiple
inheritance of Week 1, Unit 1, used for real.

Structurally these are the duck-typed contracts of Week 1, Unit 2. `typing.Protocol` is
their structural twin; we use `ABC` because that is the mechanism Week 1 practices for
SOLID, and because an explicit base class makes the contract tests trivial to write.

Interface segregation is the point: `DescentOptimizer` asks only for an `Objective`,
`SGD` only for a `BatchObjective`, `NewtonDirection` adds `TwiceDifferentiable`, and
`GaussNewton` depends on `LeastSquaresProblem` — which is deliberately *not* an
`Objective`. No implementation is ever forced to raise `NotImplementedError` to satisfy
a method it does not have.
"""

from abc import ABC, abstractmethod

from .results import OptimizeResult, StepEvent
from .types import Index, Mat, Vec

# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #


class Objective(ABC):
    """A differentiable scalar function of a parameter vector.

    The two methods must be consistent: `gradient(x)` is the gradient of `value` at `x`.
    `check_gradient` (day 1) is what proves it, and the `Objective` contract test runs
    that check over every implementation.
    """

    @abstractmethod
    def value(self, x: Vec) -> float:
        """f(x)."""

    @abstractmethod
    def gradient(self, x: Vec) -> Vec:
        """∇f(x), with the same shape as `x`."""


class TwiceDifferentiable(ABC):
    """An objective that can also produce its Hessian.

    Mixed into an `Objective`; required by `NewtonDirection`.
    """

    @abstractmethod
    def hessian(self, x: Vec) -> Mat:
        """∇²f(x), symmetric of shape (n, n)."""


class BatchObjective(ABC):
    """A finite-sum objective f(x) = (1/n) Σ fᵢ(x) whose terms can be sampled.

    Mixed into an `Objective`; required by `SGD` and `Adam`. The batch gradient must be
    an unbiased estimate of the full gradient: averaging `batch_gradient` over all
    indices must reproduce `gradient` exactly.
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """n, the number of terms in the sum."""

    @abstractmethod
    def batch_gradient(self, x: Vec, idx: Index) -> Vec:
        """The gradient of the mean of the terms selected by `idx`."""


class PointwiseLoss(ABC):
    """A per-sample loss φ(z, y) and its first two derivatives with respect to z.

    `z` is the linear predictor Xw, `y` the observation. Vectorized: all three take and
    return arrays of the same shape. This is the object `GLMLoss` is injected with —
    one loss class per likelihood, one GLM class for all of them.
    """

    @abstractmethod
    def value(self, z: Vec, y: Vec) -> Vec:
        """φ(z, y), elementwise."""

    @abstractmethod
    def d1(self, z: Vec, y: Vec) -> Vec:
        """∂φ/∂z, elementwise."""

    @abstractmethod
    def d2(self, z: Vec, y: Vec) -> Vec:
        """∂²φ/∂z², elementwise. Non-negative for the convex losses of this course."""


class LeastSquaresProblem(ABC):
    """A nonlinear least-squares problem: minimize ½‖r(x)‖².

    Deliberately NOT an `Objective`. Gauss-Newton and Levenberg-Marquardt need the
    residuals and the Jacobian separately — collapsing them into value/gradient would
    throw away the structure that makes those methods work. This is interface
    segregation as a design decision, not an accident.
    """

    @abstractmethod
    def residuals(self, x: Vec) -> Vec:
        """r(x), of shape (m,)."""

    @abstractmethod
    def jacobian(self, x: Vec) -> Mat:
        """J(x) = ∂r/∂x, of shape (m, n)."""


# --------------------------------------------------------------------------- #
# Regularization
# --------------------------------------------------------------------------- #


class Regularizer(ABC):
    """A penalty r(w), separate from the data-fit loss.

    `gradient` is allowed to raise for a non-smooth penalty (L1 does): that is exactly
    why `prox` exists and why proximal methods ask only for `prox`. The contract test
    checks the defining identity of the proximal operator,

        prox_{t·r}(v) = argmin_w  ½‖w − v‖² + t·r(w)

    numerically on small examples.
    """

    @abstractmethod
    def value(self, w: Vec) -> float:
        """r(w)."""

    @abstractmethod
    def gradient(self, w: Vec) -> Vec:
        """∇r(w). May raise `NotImplementedError` if r is not differentiable."""

    @abstractmethod
    def prox(self, w: Vec, t: float) -> Vec:
        """The proximal operator of t·r evaluated at w."""


# --------------------------------------------------------------------------- #
# The pieces a descent loop is built from
# --------------------------------------------------------------------------- #


class DirectionRule(ABC):
    """Chooses the search direction at the current point.

    May be stateful (`HeavyBall` keeps the previous step). A direction must be a descent
    direction: gᵀd < 0 whenever g ≠ 0.
    """

    @abstractmethod
    def direction(self, objective: Objective, x: Vec, g: Vec) -> Vec:
        """The search direction at `x`, where the gradient is `g`."""


class LineSearch(ABC):
    """Chooses how far to move along a given direction."""

    @abstractmethod
    def step(self, objective: Objective, x: Vec, g: Vec, d: Vec) -> float:
        """The step length α > 0 to take along `d`.

        Raises `LineSearchFailed` if no acceptable step was found.
        """


class StoppingCriterion(ABC):
    """Decides when the loop ends, from the event of the step just taken."""

    @abstractmethod
    def should_stop(self, event: StepEvent) -> bool:
        """True if the optimizer should stop now."""


class Observer(ABC):
    """Watches the loop without influencing it. `History` is the one you write."""

    @abstractmethod
    def on_step(self, event: StepEvent) -> None:
        """Called once per iteration."""


# --------------------------------------------------------------------------- #
# Linear algebra and the optimizers themselves
# --------------------------------------------------------------------------- #


class LinearSolver(ABC):
    """Solves A x = b for a symmetric positive definite A.

    One abstraction, reused by `NewtonDirection`, `GaussNewton` and
    `LevenbergMarquardt` — which is why a conjugate-gradient solver could be dropped in
    later without editing any of them.
    """

    @abstractmethod
    def solve(self, A: Mat, b: Vec) -> Vec:
        """x such that A x = b.

        Raises `NotPositiveDefiniteError` if A is not positive definite.
        """


class Optimizer(ABC):
    """Minimizes an objective from a starting point.

    The argument is typed `object` because the family is wider than `Objective`:
    `GaussNewton` and `LevenbergMarquardt` minimize a `LeastSquaresProblem`, and `SGD`
    needs a `BatchObjective`. Each implementation narrows this in its own signature and
    documents what it requires.
    """

    @abstractmethod
    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Run the optimization from `x0` and report the outcome."""
