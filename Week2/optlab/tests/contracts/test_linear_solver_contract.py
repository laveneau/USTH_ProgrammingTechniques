"""Any `ILinearSolver` must solve SPD systems and refuse non-SPD ones.

Only `CholeskySolver` exists this week. The point of writing the test against the
interface is that a conjugate-gradient solver added later would be held to exactly the
same contract, with nothing here changing.
"""

import numpy as np
import pytest

from optlab.errors import NotPositiveDefiniteError
from optlab.interfaces import ILinearSolver
from optlab.linalg import CholeskySolver

ALL_SOLVERS = [("cholesky", CholeskySolver())]


@pytest.mark.contract
@pytest.mark.day4
@pytest.mark.parametrize("name,solver", ALL_SOLVERS, ids=[n for n, _ in ALL_SOLVERS])
def test_solves_an_spd_system(name: str, solver: ILinearSolver) -> None:
    A = np.array([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
    b = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(A @ solver.solve(A, b), b, rtol=1e-10, atol=1e-12)


@pytest.mark.contract
@pytest.mark.day4
@pytest.mark.parametrize("name,solver", ALL_SOLVERS, ids=[n for n, _ in ALL_SOLVERS])
def test_matches_the_numpy_oracle(name: str, solver: ILinearSolver, rng) -> None:
    M = rng.normal(size=(5, 5))
    A = M @ M.T + 5 * np.eye(5)
    b = rng.normal(size=5)
    np.testing.assert_allclose(solver.solve(A, b), np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)


@pytest.mark.contract
@pytest.mark.day4
@pytest.mark.parametrize("name,solver", ALL_SOLVERS, ids=[n for n, _ in ALL_SOLVERS])
def test_rejects_a_non_positive_definite_matrix(name: str, solver: ILinearSolver) -> None:
    """[[1, 2], [2, 1]] has a negative eigenvalue; the factorization must not pretend."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(NotPositiveDefiniteError):
        solver.solve(A, np.array([1.0, 1.0]))
