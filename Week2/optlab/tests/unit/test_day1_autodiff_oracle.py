"""[DAY 1] Every gradient, checked a second time against exact automatic differentiation.

`check_gradient` compares against finite differences, which bottom out near 1e-11 - day 1's
U-curve shows why. `autodiff_gradient` has no step size and no floor, so it disagrees with a
correct analytic gradient only at the last bit or two. Two oracles catch different mistakes:
finite differences find sign and scale errors, autodiff pins down the rest.

The losses are written here a second time, over `Tensor` instead of numpy, straight from the
formulas in Lecture 1 Sec. 3. That duplication is the point - an oracle that reused
`GLMLoss` would agree with it even when both are wrong.
"""

import numpy as np
import pytest

from optlab.autodiff import Tensor, autodiff_gradient, exp, log
from optlab.losses import PoissonNLL
from optlab.problems import GLMLoss, linear_regression, logistic_regression
from optlab.types import Mat, Vec

pytestmark = pytest.mark.day1


@pytest.fixture
def data() -> tuple[Mat, Vec, Vec, Vec]:
    """One design matrix, three kinds of response - the three rows of the Sec. 3.4 table."""
    rng = np.random.default_rng(0)
    n, p = 60, 4
    X = rng.normal(size=(n, p))
    y_linear = X @ rng.normal(size=p) + 0.1 * rng.normal(size=n)
    y_binary = (rng.random(n) < 0.5).astype(float)
    y_count = rng.poisson(1.0, size=n).astype(float)
    return X, y_linear, y_binary, y_count


def test_squared_error_gradient_matches_autodiff(data) -> None:
    X, y, _, _ = data
    n = X.shape[0]

    def loss(w: Tensor) -> Tensor:
        return (0.5 * ((X @ w - y) ** 2)).sum() / n

    w = np.array([0.4, -1.1, 0.2, 0.7])
    np.testing.assert_allclose(
        linear_regression(X, y).gradient(w), autodiff_gradient(loss)(w), rtol=1e-12
    )


def test_logistic_gradient_matches_autodiff(data) -> None:
    X, _, y, _ = data
    n = X.shape[0]

    def loss(w: Tensor) -> Tensor:
        z = X @ w
        return (log(1 + exp(z)) - y * z).sum() / n

    w = np.array([0.4, -1.1, 0.2, 0.7])
    np.testing.assert_allclose(
        logistic_regression(X, y).gradient(w), autodiff_gradient(loss)(w), rtol=1e-12
    )


def test_poisson_gradient_matches_autodiff(data) -> None:
    X, _, _, y = data
    n = X.shape[0]

    def loss(w: Tensor) -> Tensor:
        z = X @ w
        return (exp(z) - y * z).sum() / n

    w = np.array([0.1, -0.3, 0.2, 0.05])
    np.testing.assert_allclose(
        GLMLoss(X, y, PoissonNLL()).gradient(w), autodiff_gradient(loss)(w), rtol=1e-12
    )


def test_autodiff_is_sharper_than_finite_differences(data) -> None:
    """The claim Lecture 1 Sec. 8 makes, as a test.

    Both oracles agree with the analytic gradient, but not to the same number of digits:
    central differences stop at about 1e-11 relative, autodiff goes to machine precision.
    """
    from optlab.numerics import numerical_gradient

    X, y, _, _ = data
    n = X.shape[0]
    problem = linear_regression(X, y)
    w = np.array([0.4, -1.1, 0.2, 0.7])

    def loss(t: Tensor) -> Tensor:
        return (0.5 * ((X @ t - y) ** 2)).sum() / n

    exact = problem.gradient(w)
    scale = np.linalg.norm(exact)
    err_fd = np.linalg.norm(numerical_gradient(problem.value, w) - exact) / scale
    err_ad = np.linalg.norm(autodiff_gradient(loss)(w) - exact) / scale

    assert err_fd < 1e-6
    assert err_ad < 1e-14
    assert err_ad < err_fd
