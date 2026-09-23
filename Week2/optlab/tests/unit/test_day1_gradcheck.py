"""[DAY 1] The tool everything else is verified with, verified itself."""

import numpy as np
import pytest

from optlab.numerics import check_gradient, numerical_gradient, numerical_jacobian

pytestmark = pytest.mark.day1


def test_numerical_gradient_on_a_known_function() -> None:
    f = lambda x: float(np.sum(x**2))
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(numerical_gradient(f, x), 2 * x, rtol=1e-6, atol=1e-8)


def test_numerical_jacobian_on_a_linear_map() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(
        numerical_jacobian(lambda x: A @ x, np.array([1.0, 1.0])), A, rtol=1e-6, atol=1e-8
    )


def test_check_gradient_accepts_a_correct_gradient() -> None:
    f = lambda x: float(np.sum(x**3))
    assert check_gradient(f, lambda x: 3 * x**2, np.array([0.5, -1.5])) < 1e-5


def test_check_gradient_rejects_a_wrong_gradient() -> None:
    """A gradient check that never fails is not checking anything."""
    f = lambda x: float(np.sum(x**3))
    with pytest.raises(AssertionError):
        check_gradient(f, lambda x: 2 * x**2, np.array([0.5, -1.5]))
