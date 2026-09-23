"""Liskov substitution, made executable: every `IObjective` must satisfy the same test.

Add your implementations to `OBJECTIVES` as you write them. If a subclass cannot pass a
test its base class passes, it is not substitutable — and the design, not the test, is
what needs changing.

Each entry is a *factory*, not an instance, so that a constructor which is still a stub
fails its own test rather than breaking collection of the whole suite.
"""

from collections.abc import Callable

import numpy as np
import pytest

from optlab.interfaces import IObjective
from optlab.numerics import check_gradient
from optlab.problems import Quadratic, Rosenbrock, linear_regression, logistic_regression

Case = tuple[Callable[[], IObjective], Callable[[], np.ndarray]]


def _design_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    return X, rng.normal(size=20), (rng.normal(size=20) > 0).astype(float)


OBJECTIVES: dict[str, Case] = {
    "quadratic": (
        lambda: Quadratic.ill_conditioned(n=4, kappa=10.0),
        lambda: np.ones(4),
    ),
    "rosenbrock": (
        Rosenbrock,
        lambda: np.array([-1.2, 1.0]),
    ),
    "linear": (
        lambda: linear_regression(_design_matrix()[0], _design_matrix()[1]),
        lambda: np.array([0.3, -0.2, 0.1]),
    ),
    "logistic": (
        lambda: logistic_regression(_design_matrix()[0], _design_matrix()[2]),
        lambda: np.array([0.3, -0.2, 0.1]),
    ),
}

CASES = list(OBJECTIVES.items())
IDS = list(OBJECTIVES)


@pytest.mark.contract
@pytest.mark.day1
@pytest.mark.parametrize("name,case", CASES, ids=IDS)
def test_value_returns_a_finite_scalar(name: str, case: Case) -> None:
    make_objective, make_x = case
    value = make_objective().value(make_x())
    assert isinstance(value, float)
    assert np.isfinite(value)


@pytest.mark.contract
@pytest.mark.day1
@pytest.mark.parametrize("name,case", CASES, ids=IDS)
def test_gradient_has_the_shape_of_x(name: str, case: Case) -> None:
    make_objective, make_x = case
    x = make_x()
    assert make_objective().gradient(x).shape == x.shape


@pytest.mark.contract
@pytest.mark.day1
@pytest.mark.parametrize("name,case", CASES, ids=IDS)
def test_gradient_agrees_with_finite_differences(name: str, case: Case) -> None:
    """The contract that matters: `gradient` really is the derivative of `value`."""
    make_objective, make_x = case
    objective = make_objective()
    check_gradient(objective.value, objective.gradient, make_x(), tol=1e-5)
