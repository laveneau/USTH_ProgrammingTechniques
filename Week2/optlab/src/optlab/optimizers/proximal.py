"""[DAY 6] Optimizing a sum of a smooth part and a non-smooth one.

Gradient descent cannot minimize ‖w‖₁: it is not differentiable where the solution
wants to be. The proximal gradient method takes a gradient step on the smooth part and
then applies the prox of the penalty, which for L1 is the soft-threshold — and the
threshold is what sets coefficients to exactly zero rather than merely small.

This optimizer needs `Regularizer.prox` and nothing else. It never calls `gradient` on
the penalty, which is why `L1.gradient` is free to raise.
"""

import numpy as np

from ..interfaces import Objective, Observer, Optimizer, Regularizer, StoppingCriterion
from ..results import OptimizeResult
from ..types import Vec


class ProximalGradient(Optimizer):
    """ISTA:  w⁺ = prox_{α·r}(w − α·∇f(w)),  requiring α <= 1/L.

    With `accelerated=True`, FISTA: the same step with Nesterov momentum on the
    extrapolation point, improving O(1/k) to O(1/k²).

    Lasso is `ProximalGradient(smooth=GLMLoss(...), reg=L1(lam))`. Note what that reuses:
    `GLMLoss` unchanged, from day 1. A whole new estimator, no edit to an existing file.
    """

    def __init__(
        self,
        smooth: Objective,
        reg: Regularizer,
        step: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        accelerated: bool = False,
        observers: list[Observer] | None = None,
    ) -> None:
        self.smooth = smooth
        self.reg = reg
        self.step = step
        self.max_iter = max_iter
        self.tol = tol
        self.accelerated = accelerated
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """`objective` is ignored: the smooth part and the penalty arrive in the constructor.

        Convergence is measured on the proximal-gradient residual ‖w⁺ − w‖/α, not on a
        gradient norm — the gradient of the full objective does not exist at the solution,
        which is the whole point.
        """
        raise NotImplementedError("[DAY 6] lab 1")
