"""[DAY 1] The hand-computed examples from the lecture, turned into tests.

Values you can check on paper are worth more than values you cannot: when one of these
fails you know exactly which term is wrong.
"""

import numpy as np
import pytest

from optlab.losses import LogisticNLL, SquaredError
from optlab.problems import linear_regression, logistic_regression

pytestmark = pytest.mark.day1


def test_squared_error_hand_value(hand_X, hand_y_linear) -> None:
    """X = [[1], [2]], y = (2, 4), w = 0  ->  value 5."""
    loss = linear_regression(hand_X, hand_y_linear)
    assert loss.value(np.zeros(1)) == pytest.approx(5.0)


def test_squared_error_hand_gradient(hand_X, hand_y_linear) -> None:
    """Same point  ->  gradient -5."""
    loss = linear_regression(hand_X, hand_y_linear)
    np.testing.assert_allclose(loss.gradient(np.zeros(1)), np.array([-5.0]))


def test_logistic_hand_value(hand_X, hand_y_logistic) -> None:
    """y = (0, 1), w = 0  ->  value log 2."""
    loss = logistic_regression(hand_X, hand_y_logistic)
    assert loss.value(np.zeros(1)) == pytest.approx(np.log(2.0))


def test_logistic_hand_gradient(hand_X, hand_y_logistic) -> None:
    """Same point  ->  gradient -0.25."""
    loss = logistic_regression(hand_X, hand_y_logistic)
    np.testing.assert_allclose(loss.gradient(np.zeros(1)), np.array([-0.25]))


def test_logistic_does_not_overflow() -> None:
    """softplus(1000) is 1000, not inf. Computing log(1 + exp(z)) naively fails here."""
    value = LogisticNLL().value(np.array([1000.0]), np.array([0.0]))
    assert np.isfinite(value).all()
    assert value[0] == pytest.approx(1000.0)


def test_squared_error_is_the_gaussian_nll_up_to_constants() -> None:
    z, y = np.array([1.0, -1.0]), np.array([0.5, 0.5])
    np.testing.assert_allclose(SquaredError().value(z, y), 0.5 * (z - y) ** 2)
