"""Every `IBatchObjective` must agree with the `IObjective` it is mixed into.

The contract is one identity — the batch gradient over *all* the indices is the full
gradient — plus the unbiasedness that identity implies. It is what licenses SGD to use a
sample in place of the real thing, and it is the first thing to check when a stochastic
run drifts somewhere the deterministic one does not.

Add your implementations to `BATCH_OBJECTIVES` as you write them. Each entry is a
*factory*, so a constructor that is still a stub fails its own test rather than breaking
collection of the whole suite.
"""

from collections.abc import Callable

import numpy as np
import pytest

from optlab.interfaces import IBatchObjective
from optlab.problems import linear_regression, logistic_regression

N_SAMPLES = 24


def _design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(N_SAMPLES, 3))
    return X, rng.normal(size=N_SAMPLES), (rng.random(N_SAMPLES) > 0.5).astype(float)


BATCH_OBJECTIVES: dict[str, Callable[[], IBatchObjective]] = {
    "linear": lambda: linear_regression(_design()[0], _design()[1]),
    "logistic": lambda: logistic_regression(_design()[0], _design()[2]),
}

W = np.array([0.3, -0.2, 0.1])
IDS = list(BATCH_OBJECTIVES)


@pytest.mark.contract
@pytest.mark.day3
@pytest.mark.parametrize("name", IDS)
def test_n_samples_is_a_positive_int(name: str) -> None:
    n = BATCH_OBJECTIVES[name]().n_samples
    assert isinstance(n, int)
    assert n == N_SAMPLES


@pytest.mark.contract
@pytest.mark.day3
@pytest.mark.parametrize("name", IDS)
def test_batch_gradient_has_the_shape_of_w(name: str) -> None:
    problem = BATCH_OBJECTIVES[name]()
    assert problem.batch_gradient(W, np.array([0, 3, 7])).shape == W.shape


@pytest.mark.contract
@pytest.mark.day3
@pytest.mark.parametrize("name", IDS)
def test_the_whole_batch_is_the_full_gradient(name: str) -> None:
    """The defining identity, and the one the docstring of `IBatchObjective` states."""
    problem = BATCH_OBJECTIVES[name]()
    everything = np.arange(problem.n_samples)
    full = problem.gradient(W)          # type: ignore[attr-defined]
    assert problem.batch_gradient(W, everything) == pytest.approx(full)


@pytest.mark.contract
@pytest.mark.day3
@pytest.mark.parametrize("name", IDS)
def test_the_estimate_is_unbiased(name: str) -> None:
    """Averaging over every singleton batch reproduces the full gradient.

    Where the argument uses uniform sampling: this mean weights every sample equally,
    which is only the right weighting if every sample was equally likely to be drawn.
    """
    problem = BATCH_OBJECTIVES[name]()
    singles = np.array(
        [problem.batch_gradient(W, np.array([i])) for i in range(problem.n_samples)]
    )
    assert singles.mean(axis=0) == pytest.approx(problem.gradient(W))  # type: ignore[attr-defined]


@pytest.mark.contract
@pytest.mark.day3
@pytest.mark.parametrize("name", IDS)
def test_the_order_of_the_indices_does_not_matter(name: str) -> None:
    """A batch is a set. Shuffling it may change the last bits, never the answer."""
    problem = BATCH_OBJECTIVES[name]()
    idx = np.array([1, 4, 9, 2, 7])
    assert problem.batch_gradient(W, idx) == pytest.approx(
        problem.batch_gradient(W, idx[::-1])
    )


@pytest.mark.contract
@pytest.mark.day3
@pytest.mark.parametrize("name", IDS)
def test_a_repeated_batch_weights_its_samples_twice(name: str) -> None:
    """The divisor is `len(idx)`, not the number of distinct samples in it.

    Sampling with replacement is legitimate — notebook 3 compares it with reshuffling —
    so a duplicated index must count twice rather than being quietly deduplicated.
    """
    problem = BATCH_OBJECTIVES[name]()
    one = problem.batch_gradient(W, np.array([5]))
    assert problem.batch_gradient(W, np.array([5, 5])) == pytest.approx(one)
    pair = problem.batch_gradient(W, np.array([5, 6]))
    lopsided = problem.batch_gradient(W, np.array([5, 5, 6]))
    assert not np.allclose(pair, lopsided)
