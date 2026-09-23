"""[DAY 6] Soft-thresholding, by hand."""

import numpy as np
import pytest

from optlab.regularizers import L1

pytestmark = pytest.mark.day6


def test_soft_threshold_hand_example() -> None:
    """v = (3, -0.4, 0.1) at t = 0.5 with lambda = 1  ->  (2.5, 0, 0).

    Two of three coefficients become exactly zero, not merely small. That is the whole
    argument for L1 over ridge, in one line of arithmetic.
    """
    np.testing.assert_allclose(
        L1(lam=1.0).prox(np.array([3.0, -0.4, 0.1]), 0.5), np.array([2.5, 0.0, 0.0])
    )


def test_soft_threshold_is_odd() -> None:
    v = np.array([3.0, -0.4, 0.1, -2.0])
    reg = L1(lam=1.0)
    np.testing.assert_allclose(reg.prox(-v, 0.5), -reg.prox(v, 0.5))


def test_large_threshold_zeroes_everything() -> None:
    """Past lambda·t = max|v| the whole solution collapses to zero: the end of the path."""
    np.testing.assert_allclose(
        L1(lam=10.0).prox(np.array([3.0, -0.4, 0.1]), 1.0), np.zeros(3)
    )
