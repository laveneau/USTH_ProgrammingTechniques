"""Exceptions. PROVIDED — do not edit."""


class OptlabError(Exception):
    """Base class for every error raised by optlab."""


class NotPositiveDefiniteError(OptlabError):
    """A Cholesky factorization was attempted on a matrix that is not positive definite.

    Raised by `CholeskySolver`. Geometrically the quadratic model has a direction of
    non-positive curvature; statistically the Fisher information is singular or
    indefinite, so the parameters are not identified from the data.
    """


class LineSearchFailed(OptlabError):
    """A line search could not find an acceptable step.

    Raised when the backtracking budget is exhausted. Optimizers are expected to catch
    this and return an `OptimizeResult` with `converged=False`, not to propagate it.
    """
