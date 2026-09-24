"""Every `IRegularizer` must satisfy the defining identity of its proximal operator:

    prox_{t·r}(v) = argmin_w  ½‖w − v‖² + t·r(w)

Checked numerically: the returned point must beat its neighbours on that objective.
This catches a soft-threshold with the wrong sign or a missing factor of t, which a
convergence test would only reveal much later and much more confusingly.
"""

import numpy as np
import pytest

from optlab.interfaces import IRegularizer
from optlab.regularizers import L1, L2, ElasticNet, NoRegularizer

DAY4 = [("none", NoRegularizer()), ("l2", L2(lam=0.7))]
DAY6 = [("l1", L1(lam=0.5)), ("elasticnet", ElasticNet(lam=0.5, alpha=0.6))]


def _prox_objective(w: np.ndarray, v: np.ndarray, t: float, reg: IRegularizer) -> float:
    return 0.5 * float(np.sum((w - v) ** 2)) + t * reg.value(w)


def _assert_prox_is_optimal(reg: IRegularizer, v: np.ndarray, t: float, rng) -> None:
    w = reg.prox(v, t)
    best = _prox_objective(w, v, t, reg)
    for _ in range(200):
        candidate = w + rng.normal(scale=0.05, size=w.shape)
        assert _prox_objective(candidate, v, t, reg) >= best - 1e-9


@pytest.mark.contract
@pytest.mark.day4
@pytest.mark.parametrize("name,reg", DAY4, ids=[n for n, _ in DAY4])
def test_prox_minimizes_its_defining_objective(name: str, reg: IRegularizer, rng) -> None:
    _assert_prox_is_optimal(reg, np.array([3.0, -0.4, 0.1, 0.0]), 0.5, rng)


@pytest.mark.contract
@pytest.mark.day6
@pytest.mark.parametrize("name,reg", DAY6, ids=[n for n, _ in DAY6])
def test_non_smooth_prox_minimizes_its_defining_objective(
    name: str, reg: IRegularizer, rng
) -> None:
    _assert_prox_is_optimal(reg, np.array([3.0, -0.4, 0.1, 0.0]), 0.5, rng)


@pytest.mark.contract
@pytest.mark.day4
@pytest.mark.parametrize("name,reg", DAY4, ids=[n for n, _ in DAY4])
def test_prox_with_t_zero_is_the_identity(name: str, reg: IRegularizer) -> None:
    v = np.array([3.0, -0.4, 0.1, 0.0])
    np.testing.assert_allclose(reg.prox(v, 0.0), v, rtol=1e-12, atol=1e-14)


@pytest.mark.contract
@pytest.mark.day6
@pytest.mark.parametrize("name,reg", DAY6, ids=[n for n, _ in DAY6])
def test_non_smooth_prox_with_t_zero_is_the_identity(name: str, reg: IRegularizer) -> None:
    """Split from the day-4 case on purpose: `prox` at t = 0 is the identity for every
    regularizer, but `L1` and `ElasticNet` are not written until day 6, and a day-4
    student must be able to make `-m day4` green with day-4 work alone."""
    v = np.array([3.0, -0.4, 0.1, 0.0])
    np.testing.assert_allclose(reg.prox(v, 0.0), v, rtol=1e-12, atol=1e-14)


@pytest.mark.contract
@pytest.mark.day6
def test_l1_gradient_refuses_rather_than_lying() -> None:
    """Not a gap in the implementation — the contract. L1 has no gradient at zero."""
    with pytest.raises(NotImplementedError):
        L1(lam=1.0).gradient(np.zeros(3))
