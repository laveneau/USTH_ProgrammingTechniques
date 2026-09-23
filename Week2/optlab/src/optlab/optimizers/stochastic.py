"""[DAY 3] Training on a sample of the gradient — how modern machine learning actually fits.

These are their own `IOptimizer`s rather than `IDirectionRule`s: the loop is different
(epochs, shuffling, no line search), so forcing them into `DescentOptimizer` would mean
editing it. A new implementation of an existing interface is the open/closed answer.

Both depend only on `IBatchObjective`. Neither asks for a Hessian, a line search, or a
full gradient.
"""

import numpy as np

from ..interfaces import IBatchObjective, IObserver, IOptimizer, IStoppingCriterion
from ..results import OptimizeResult, StepEvent
from ..types import Vec


class SGD(IOptimizer):
    """Mini-batch stochastic gradient descent.

    The random generator is injected, never created inside: that is what makes a run
    reproducible from a seed, and it is dependency inversion applied to randomness.

    With `batch_size = n` this must reduce exactly to gradient descent with a fixed
    step — there is a test for it.

    A constant step converges only to a noise floor proportional to α; a decaying
    schedule satisfying the Robbins-Monro conditions converges properly. Day 3 asks you
    to see both.
    """

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.01,
        n_epochs: int = 50,
        momentum: float = 0.0,
        lr_decay: float = 0.0,
        rng: np.random.Generator | None = None,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.rng = rng if rng is not None else np.random.default_rng()
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `IBatchObjective`."""
        raise NotImplementedError("[DAY 3] lab 2")


class Adam(IOptimizer):
    """Adaptive moment estimation: a per-coordinate step size from the running moments.

    Keeps m (first moment) and v (second), both bias-corrected because they start at
    zero and would otherwise be biased toward it for the first several steps.

    What it really does is auto-standardize the per-coordinate scale — the optimizer-side
    answer to the conditioning problem that day 1 diagnosed in the data. On the ill-scaled
    (1 vs 1000) problem it needs far fewer epochs than SGD; that contrast is the lab.

    Its very first step is approximately lr·sign(g), which is a good test.
    """

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.001,
        n_epochs: int = 50,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        rng: np.random.Generator | None = None,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `IBatchObjective`."""
        raise NotImplementedError("[DAY 3] lab 3")
