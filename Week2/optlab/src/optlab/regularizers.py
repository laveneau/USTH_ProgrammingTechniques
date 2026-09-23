"""Penalties, kept strictly separate from the data-fit loss.

Under the MAP reading a penalty is a prior: ½λ‖w‖² is a Gaussian prior, λ‖w‖₁ a Laplace
one. The Laplace prior has a spike at zero, which is why L1 produces exact zeros and
ridge does not.
"""

import numpy as np

from .interfaces import IRegularizer
from .types import Vec


class NoRegularizer(IRegularizer):
    """[DAY 4] r(w) = 0. The neutral element, so unregularized fitting is not a special case."""

    def value(self, w: Vec) -> float:
        raise NotImplementedError("[DAY 4] lab 4")

    def gradient(self, w: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 4")

    def prox(self, w: Vec, t: float) -> Vec:
        raise NotImplementedError("[DAY 4] lab 4")


class L2(IRegularizer):
    """[DAY 4] r(w) = ½λ‖w‖². Ridge. Smooth, so ordinary descent and Newton handle it.

    Its prox is the shrinkage w / (1 + λt). Adding λI to the Hessian also makes it
    unconditionally positive definite, so Cholesky can no longer fail — the same trick
    LM uses on day 5.
    """

    def __init__(self, lam: float = 1.0) -> None:
        self.lam = lam

    def value(self, w: Vec) -> float:
        raise NotImplementedError("[DAY 4] lab 4")

    def gradient(self, w: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 4")

    def prox(self, w: Vec, t: float) -> Vec:
        raise NotImplementedError("[DAY 4] lab 4")


class L1(IRegularizer):
    """[DAY 6] r(w) = λ‖w‖₁. Lasso. NOT differentiable at zero.

    `gradient` must raise `NotImplementedError` — that is not a gap, it is the contract.
    The kink at zero is precisely what pins coefficients to exactly zero, and the way
    past it is `prox`, the soft-threshold sign(v)·max(|v| − λt, 0).
    """

    def __init__(self, lam: float = 1.0) -> None:
        self.lam = lam

    def value(self, w: Vec) -> float:
        raise NotImplementedError("[DAY 6] lab 1")

    def gradient(self, w: Vec) -> Vec:
        raise NotImplementedError("L1 is not differentiable at 0; use prox()")

    def prox(self, w: Vec, t: float) -> Vec:
        raise NotImplementedError("[DAY 6] lab 1")


class ElasticNet(IRegularizer):
    """[DAY 6] λ(α‖w‖₁ + ½(1−α)‖w‖²). Sparsity plus the stability of ridge.

    Its prox composes the two: soft-threshold, then shrink.
    """

    def __init__(self, lam: float = 1.0, alpha: float = 0.5) -> None:
        self.lam = lam
        self.alpha = alpha

    def value(self, w: Vec) -> float:
        raise NotImplementedError("[DAY 6] lab 1")

    def gradient(self, w: Vec) -> Vec:
        raise NotImplementedError("not differentiable for alpha > 0; use prox()")

    def prox(self, w: Vec, t: float) -> Vec:
        raise NotImplementedError("[DAY 6] lab 1")
