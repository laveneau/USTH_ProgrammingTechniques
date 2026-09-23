"""Result and event records. PROVIDED — do not edit.

Both are frozen dataclasses: the `@dataclass` first met in the Week 1 autodiff project.
Frozen because an observer must not be able to alter the history it is handed.
"""

from dataclasses import dataclass

from .types import Vec


@dataclass(frozen=True)
class StepEvent:
    """One iteration of an optimizer, handed to every `IObserver` and `IStoppingCriterion`.

    `step_size` is the length actually taken along `direction` (1.0 for methods that do
    not scale their step). `grad_norm` is the Euclidean norm of the gradient at `x`.
    """

    iteration: int
    x: Vec
    value: float
    grad_norm: float
    step_size: float


@dataclass(frozen=True)
class OptimizeResult:
    """What every `IOptimizer.minimize` returns.

    `converged` is True only if a stopping criterion fired on a genuine convergence
    test. Hitting `MaxIterations`, or catching a `LineSearchFailed`, gives
    `converged=False` with the reason in `message`.
    """

    x: Vec
    value: float
    grad_norm: float
    iterations: int
    converged: bool
    message: str = ""
