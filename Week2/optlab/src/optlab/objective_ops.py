"""[DAY 4] Combining an objective with a regularizer."""

from .interfaces import IObjective, IRegularizer, ITwiceDifferentiable
from .types import Mat, Vec


class RegularizedObjective(IObjective, ITwiceDifferentiable):
    """f(w) + r(w), presented as a single `IObjective`.

    An adapter, and the open/closed moment of the week: ridge regression is not a new
    optimizer, a new loss, or a new anything — it is `RegularizedObjective(GLMLoss, L2)`
    handed to an optimizer you already wrote. Every method from days 2 and 3 gains a
    regularized version the moment this class exists.

    Requires an `ITwiceDifferentiable` inner objective only if you ask it for a Hessian.
    """

    def __init__(self, objective: IObjective, regularizer: IRegularizer) -> None:
        self.objective = objective
        self.regularizer = regularizer

    def value(self, w: Vec) -> float:
        raise NotImplementedError("[DAY 4] lab 4")

    def gradient(self, w: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 4")

    def hessian(self, w: Vec) -> Mat:
        """For L2 this adds λI, which guarantees positive definiteness."""
        raise NotImplementedError("[DAY 4] lab 4")
