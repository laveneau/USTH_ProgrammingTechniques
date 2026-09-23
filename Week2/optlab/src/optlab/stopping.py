"""[DAY 2] When the loop ends."""

from .interfaces import StoppingCriterion
from .results import StepEvent


class GradientNormBelow(StoppingCriterion):
    """Stop once ‖∇f‖ <= tol. The only criterion here that means genuine convergence."""

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def should_stop(self, event: StepEvent) -> bool:
        raise NotImplementedError("[DAY 2] lab 1")


class MaxIterations(StoppingCriterion):
    """Stop after a budget. Firing this means `converged=False`, not success."""

    def __init__(self, max_iter: int = 1000) -> None:
        self.max_iter = max_iter

    def should_stop(self, event: StepEvent) -> bool:
        raise NotImplementedError("[DAY 2] lab 1")


class AnyOf(StoppingCriterion):
    """Stop as soon as any of the wrapped criteria fires.

    A composite, so the loop still sees exactly one `StoppingCriterion` no matter how
    many conditions you combine.
    """

    def __init__(self, *criteria: StoppingCriterion) -> None:
        self.criteria = criteria

    def should_stop(self, event: StepEvent) -> bool:
        raise NotImplementedError("[DAY 2] lab 1")
