"""Every `IPointwiseLoss` must have consistent derivatives, whatever the likelihood."""

import numpy as np
import pytest

from optlab.interfaces import IPointwiseLoss
from optlab.losses import Huber, LogisticNLL, PoissonNLL, SquaredError

SMOOTH = [("squared", SquaredError()), ("logistic", LogisticNLL())]
DAY6 = [("huber", Huber(delta=1.0)), ("poisson", PoissonNLL())]


def _numeric_d(f, z: np.ndarray, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return (f(z + h, y) - f(z - h, y)) / (2 * h)


@pytest.mark.contract
@pytest.mark.day1
@pytest.mark.parametrize("name,loss", SMOOTH, ids=[n for n, _ in SMOOTH])
def test_d1_is_the_derivative_of_value(name: str, loss: IPointwiseLoss) -> None:
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(loss.d1(z, y), _numeric_d(loss.value, z, y), rtol=1e-5, atol=1e-7)


@pytest.mark.contract
@pytest.mark.day1
@pytest.mark.parametrize("name,loss", SMOOTH, ids=[n for n, _ in SMOOTH])
def test_d2_is_the_derivative_of_d1(name: str, loss: IPointwiseLoss) -> None:
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(loss.d2(z, y), _numeric_d(loss.d1, z, y), rtol=1e-5, atol=1e-7)


@pytest.mark.contract
@pytest.mark.day1
@pytest.mark.parametrize("name,loss", SMOOTH, ids=[n for n, _ in SMOOTH])
def test_d2_is_non_negative(name: str, loss: IPointwiseLoss) -> None:
    """These losses are convex in z, so the second derivative cannot be negative."""
    z = np.linspace(-5, 5, 41)
    y = np.zeros_like(z)
    assert np.all(loss.d2(z, y) >= -1e-12)


@pytest.mark.contract
@pytest.mark.day6
@pytest.mark.parametrize("name,loss", DAY6, ids=[n for n, _ in DAY6])
def test_day6_losses_satisfy_the_same_contract(name: str, loss: IPointwiseLoss) -> None:
    """Written on day 6, held to the day-1 contract without a line of it changing.

    The residuals `z - y` here are -2, -1.5, 0, -1.5 and 0.5: deliberately none of them
    is +-`delta`. Huber's second derivative jumps from 1 to 0 at the kink, so it simply
    does not exist there, and a central difference straddling the jump returns 0.5 -- a
    number no correct implementation can be asked to match. Checking a derivative where
    the function has none tests the test, not the code.
    """
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = np.array([0.0, 1.0, 0.0, 2.0, 1.5])
    np.testing.assert_allclose(loss.d1(z, y), _numeric_d(loss.value, z, y), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(loss.d2(z, y), _numeric_d(loss.d1, z, y), rtol=1e-4, atol=1e-6)
