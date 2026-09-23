"""Pointwise losses: one class per likelihood.

Each is the negative log-likelihood of a distribution, up to constants:
squared error ↔ Gaussian, logistic ↔ Bernoulli, Poisson ↔ Poisson counts.
Huber is not a likelihood but a robustification of the Gaussian one.
"""

import numpy as np

from .interfaces import PointwiseLoss
from .types import Vec


class SquaredError(PointwiseLoss):
    """[DAY 1] φ(z, y) = ½(z − y)². The Gaussian negative log-likelihood."""

    def value(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 1] lab 2")

    def d1(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 1] lab 2")

    def d2(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 1] lab 2")


class LogisticNLL(PointwiseLoss):
    """[DAY 1] φ(z, y) = log(1 + e^z) − y·z, for y ∈ {0, 1}.

    Watch the overflow: `log(1 + exp(1000))` is `inf` computed naively and `1000`
    computed properly. Use the stable form of softplus. The trap is on the day-1 slide
    and in the tests.
    """

    def value(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 1] lab 2")

    def d1(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 1] lab 2")

    def d2(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 1] lab 2")


class Huber(PointwiseLoss):
    """[DAY 6] Quadratic near zero, linear beyond `delta`.

    Bounded influence, so a gross outlier cannot dominate the fit — yet still smooth,
    unlike the L1 *penalty* of the same day. Note where each non-smoothness lives:
    Huber is smooth in the residual, L1 is non-smooth in the parameters.

    As `delta → ∞` it must reduce to `SquaredError`; a test checks exactly that.
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def value(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d1(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d2(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")


class PoissonNLL(PointwiseLoss):
    """[DAY 6] φ(z, y) = e^z − y·z. Counts with a log link.

    Writing this is the payoff of the open/closed principle: it is a new `PointwiseLoss`,
    so gradient descent, Newton and ridge all work on it without a line changing anywhere
    else.
    """

    def value(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d1(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d2(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")
