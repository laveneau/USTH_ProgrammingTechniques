"""[DAY 4] The factorization you can check by hand, and the one that must fail."""

import numpy as np
import pytest

from optlab.errors import NotPositiveDefiniteError
from optlab.linalg import cholesky, solve_lower, solve_upper_from_lower

pytestmark = pytest.mark.day4


def test_cholesky_hand_example() -> None:
    """[[4,2,2],[2,5,3],[2,3,6]] factors as L = [[2,0,0],[1,2,0],[1,1,2]]."""
    A = np.array([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
    expected = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 2.0]])
    np.testing.assert_allclose(cholesky(A), expected, rtol=1e-12, atol=1e-14)


def test_cholesky_reconstructs_the_matrix(rng) -> None:
    M = rng.normal(size=(6, 6))
    A = M @ M.T + 6 * np.eye(6)
    L = cholesky(A)
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-10, atol=1e-12)


def test_cholesky_is_lower_triangular(rng) -> None:
    M = rng.normal(size=(4, 4))
    L = cholesky(M @ M.T + 4 * np.eye(4))
    np.testing.assert_allclose(L, np.tril(L), atol=1e-14)


def test_cholesky_fails_on_an_indefinite_matrix() -> None:
    """[[1,2],[2,1]]: the second pivot would be L22² = -3. Report it, do not sqrt it."""
    with pytest.raises(NotPositiveDefiniteError):
        cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_triangular_solves(rng) -> None:
    M = rng.normal(size=(5, 5))
    A = M @ M.T + 5 * np.eye(5)
    L = cholesky(A)
    b = rng.normal(size=5)
    y = solve_lower(L, b)
    np.testing.assert_allclose(L @ y, b, rtol=1e-10, atol=1e-12)
    x = solve_upper_from_lower(L, y)
    np.testing.assert_allclose(A @ x, b, rtol=1e-10, atol=1e-12)
