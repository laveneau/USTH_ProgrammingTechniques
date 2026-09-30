"""The generalized linear model loss: one class for every GLM.

`GLMLoss` does not know which likelihood it is fitting - it receives an `IPointwiseLoss`
and composes with it. Linear regression, logistic regression, Poisson regression and
robust regression are all this one class with a different object injected. That is
dependency inversion, and it is why day 6 gets Poisson and Huber almost for free.

There is deliberately no `lambda` parameter here. Regularization is a separate object
(`IRegularizer`, day 4) wrapped around this one by `RegularizedObjective` - a penalty is
not part of the likelihood.
"""

import numpy as np

from ..interfaces import IBatchObjective, IObjective, IPointwiseLoss, ITwiceDifferentiable
from ..losses import LogisticNLL, SquaredError
from ..types import Index, Mat, Vec


class GLMLoss(IObjective, ITwiceDifferentiable, IBatchObjective):
    """L(w) = (1/n) sum phi(x_i^T w, y_i) for an injected pointwise loss phi.

    Three interfaces at once - the multiple inheritance of Week 1, Unit 1. Each caller
    sees only the one it needs: the descent loop an `IObjective`, `SGD` a
    `IBatchObjective`, `NewtonDirection` an `ITwiceDifferentiable`.
    """

    def __init__(self, X: Mat, y: Vec, pointwise: IPointwiseLoss) -> None:
        self.X = X
        self.y = y
        self.pointwise = pointwise

    def value(self, w: Vec) -> float:
        """[DAY 1] The mean of phi over the samples."""
        raise NotImplementedError("[DAY 1] lab 2")

    def gradient(self, w: Vec) -> Vec:
        """[DAY 1] X^T*phi'(Xw, y) / n."""
        raise NotImplementedError("[DAY 1] lab 2")

    def hessian(self, w: Vec) -> Mat:
        """[DAY 4] X^T DX / n with D = diag(phi''(Xw, y)).

        For squared error D = I and one Newton step solves the normal equations exactly.
        For the logistic loss D = diag(sigma(1-sigma)) and Newton *is* iteratively reweighted
        least squares - the algorithm statisticians use to fit GLMs. This matrix is also
        the observed Fisher information, so its inverse estimates cov(w_hat).
        """
        raise NotImplementedError("[DAY 4] lab 2")

    @property
    def n_samples(self) -> int:
        """[DAY 3] n."""
        raise NotImplementedError("[DAY 3] lab 1")

    def batch_gradient(self, w: Vec, idx: Index) -> Vec:
        """[DAY 3] The gradient over the subset `idx` only.

        Write this so that `gradient` becomes the special case over all indices, rather
        than duplicating the formula in two places.
        """
        raise NotImplementedError("[DAY 3] lab 1")


def linear_regression(X: Mat, y: Vec) -> GLMLoss:
    """[DAY 1] Least-squares regression: the Gaussian MLE."""
    raise NotImplementedError("[DAY 1] lab 2")


def logistic_regression(X: Mat, y: Vec) -> GLMLoss:
    """[DAY 1] Logistic regression: the Bernoulli MLE."""
    raise NotImplementedError("[DAY 1] lab 2")
