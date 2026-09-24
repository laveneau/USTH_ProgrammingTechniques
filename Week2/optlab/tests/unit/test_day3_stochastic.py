"""[DAY 3] The mini-batch gradient, SGD and Adam.

Read these as the specification of Labwork 3. Everything here is either a value you can
work out on paper or a property the subject and the docstrings promise.

Two of them deserve a word, because they test a *design* decision rather than a number:
`test_gradient_is_implemented_through_batch_gradient` is the refactor Exercise 1 asks
for, and `test_run_ignores_the_global_numpy_seed` is the injected-generator rule of
Exercise 2. Both would pass by accident under a copy-paste implementation and then fail
the first time somebody changed the formula or seeded numpy globally.
"""

import numpy as np
import pytest

from optlab.interfaces import IBatchObjective, IObjective
from optlab.losses import SquaredError
from optlab.observers import History
from optlab.optimizers import SGD, Adam
from optlab.problems import GLMLoss, Rosenbrock, linear_regression, logistic_regression
from optlab.types import Index, Mat, Vec

pytestmark = pytest.mark.day3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dataset(n: int = 40, d: int = 3, seed: int = 0) -> tuple[Mat, Vec, Vec]:
    """One design matrix, a continuous target and a binary one."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y_linear = X @ w + 0.1 * rng.normal(size=n)
    y_binary = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X @ w)))).astype(float)
    return X, y_linear, y_binary


class ConstantPerSample(GLMLoss):
    """Every row of X identical, so the batch gradient does not depend on *which* rows.

    That is what makes the Adam arithmetic below reproducible without also pinning down
    the order the generator happens to produce.
    """

    def __init__(self, n: int) -> None:
        super().__init__(np.ones((n, 1)), np.zeros(n), SquaredError())


class Spy(GLMLoss):
    """A `GLMLoss` that counts the calls to `batch_gradient`."""

    def __init__(self, X: Mat, y: Vec) -> None:
        super().__init__(X, y, SquaredError())
        self.calls: list[Index] = []

    def batch_gradient(self, w: Vec, idx: Index) -> Vec:
        self.calls.append(np.asarray(idx))
        return super().batch_gradient(w, idx)


class Recorder(IObjective, IBatchObjective):
    """A batch objective that remembers exactly which indices each batch contained."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.batches: list[Index] = []

    @property
    def n_samples(self) -> int:
        return self.n

    def value(self, x: Vec) -> float:
        return float(x @ x)

    def gradient(self, x: Vec) -> Vec:
        return 2.0 * x

    def batch_gradient(self, x: Vec, idx: Index) -> Vec:
        self.batches.append(np.asarray(idx).copy())
        return 2.0 * x


# --------------------------------------------------------------------------- #
# Exercise 1 — the mini-batch gradient
# --------------------------------------------------------------------------- #


def test_n_samples_counts_the_rows(hand_X: Mat, hand_y_linear: Vec) -> None:
    assert linear_regression(hand_X, hand_y_linear).n_samples == 2


def test_n_samples_is_a_plain_int(hand_X: Mat, hand_y_linear: Vec) -> None:
    """A numpy integer would compare equal and then fail `range()` in a subtle place."""
    assert isinstance(linear_regression(hand_X, hand_y_linear).n_samples, int)


def test_batch_gradient_over_one_sample_by_hand(hand_X: Mat, hand_y_linear: Vec) -> None:
    """X = [[1], [2]], y = (2, 4), w = 0.

    Row 0 alone: z = 0, φ'(z, y) = z − y = −2, so the gradient is 1·(−2)/1 = −2.
    Row 1 alone: 2·(0 − 4)/1 = −8. The full gradient, −5, is their mean.
    """
    loss = linear_regression(hand_X, hand_y_linear)
    w = np.zeros(1)
    assert loss.batch_gradient(w, np.array([0])) == pytest.approx([-2.0])
    assert loss.batch_gradient(w, np.array([1])) == pytest.approx([-8.0])
    assert loss.gradient(w) == pytest.approx([-5.0])


def test_batch_gradient_averages_over_the_batch_not_over_n() -> None:
    """The divisor is `len(idx)`.

    Dividing by n instead is the single most common mistake here, and it does not raise:
    it silently scales every step by b/n, so the run just looks slow.
    """
    X, y, _ = _dataset()
    loss = linear_regression(X, y)
    w = np.zeros(X.shape[1])
    idx = np.arange(4)
    singles = np.array([loss.batch_gradient(w, np.array([i])) for i in idx])
    assert loss.batch_gradient(w, idx) == pytest.approx(singles.mean(axis=0))


def test_batch_gradient_over_every_index_is_the_full_gradient() -> None:
    """The defining property of `IBatchObjective`, for both likelihoods."""
    X, y_linear, y_binary = _dataset()
    w = np.array([0.3, -0.2, 0.1])
    everything = np.arange(X.shape[0])
    for loss in (linear_regression(X, y_linear), logistic_regression(X, y_binary)):
        assert loss.batch_gradient(w, everything) == pytest.approx(loss.gradient(w))


def test_batch_gradient_is_unbiased() -> None:
    """Averaging the singleton gradients reproduces the full gradient exactly.

    This is the whole justification for sampling: the estimate is not systematically
    wrong in any direction. Uniform sampling is where the argument is used — the mean
    below weights every sample equally.
    """
    X, _, y = _dataset()
    loss = logistic_regression(X, y)
    w = np.array([0.5, 0.5, -0.5])
    singles = np.array([loss.batch_gradient(w, np.array([i])) for i in range(X.shape[0])])
    assert singles.mean(axis=0) == pytest.approx(loss.gradient(w))


def test_batch_gradient_has_the_shape_of_w() -> None:
    X, y, _ = _dataset()
    loss = linear_regression(X, y)
    assert loss.batch_gradient(np.zeros(3), np.array([1, 5, 7])).shape == (3,)


def test_gradient_is_implemented_through_batch_gradient() -> None:
    """Exercise 1's refactor: `Xᵀφ'/n` must appear once in the file, in `batch_gradient`.

    If `gradient` still carries its own copy of the formula this spy sees no call, and
    the two copies will disagree the first time one of them is edited.
    """
    X, y, _ = _dataset()
    spy = Spy(X, y)
    spy.gradient(np.zeros(3))
    assert len(spy.calls) == 1, "gradient() should delegate to batch_gradient()"
    assert sorted(spy.calls[0].tolist()) == list(range(X.shape[0]))


# --------------------------------------------------------------------------- #
# Exercise 2 — SGD
# --------------------------------------------------------------------------- #


def test_full_batch_sgd_is_exactly_gradient_descent() -> None:
    """The check the subject calls non-negotiable, and it is an *equality* of floats.

    With b = n there is one batch per epoch and it is the whole dataset, so the update
    is w − α∇f(w) and nothing else. A stray decay, a half-applied momentum or a wrong
    divisor all show up here as a difference far above rounding.
    """
    X, y, _ = _dataset()
    loss = linear_regression(X, y)
    w0 = np.zeros(X.shape[1])
    lr, n_epochs = 0.3, 25

    result = SGD(batch_size=X.shape[0], lr=lr, n_epochs=n_epochs,
                 rng=np.random.default_rng(0)).minimize(loss, w0)

    w = w0.copy()
    for _ in range(n_epochs):
        w = w - lr * loss.gradient(w)

    assert np.max(np.abs(result.x - w)) < 1e-12


def test_same_seed_gives_the_same_run() -> None:
    X, _, y = _dataset()
    loss = logistic_regression(X, y)
    w0 = np.zeros(X.shape[1])

    def run(seed: int) -> Vec:
        return SGD(batch_size=8, lr=0.3, n_epochs=6,
                   rng=np.random.default_rng(seed)).minimize(loss, w0).x

    assert np.array_equal(run(11), run(11))
    assert not np.allclose(run(11), run(12))


def test_run_ignores_the_global_numpy_seed() -> None:
    """The generator must be the injected one, never `np.random.*`.

    Seeding the legacy global generator differently between two runs must change
    nothing. If it does, the run is not reproducible from the seed its caller gave.
    """
    X, _, y = _dataset()
    loss = logistic_regression(X, y)
    w0 = np.zeros(X.shape[1])

    np.random.seed(0)
    first = SGD(batch_size=8, lr=0.3, n_epochs=4,
                rng=np.random.default_rng(5)).minimize(loss, w0).x
    np.random.seed(999)
    second = SGD(batch_size=8, lr=0.3, n_epochs=4,
                 rng=np.random.default_rng(5)).minimize(loss, w0).x

    assert np.array_equal(first, second)


def test_one_epoch_uses_every_sample_exactly_once() -> None:
    """Reshuffling, not sampling with replacement — and no dropped remainder.

    n = 10 with b = 4 gives blocks of 4, 4 and 2. The short block is kept: dropping it
    would quietly ignore two samples out of ten on every epoch.
    """
    recorder = Recorder(10)
    SGD(batch_size=4, lr=0.1, n_epochs=1,
        rng=np.random.default_rng(0)).minimize(recorder, np.ones(2))

    assert [len(b) for b in recorder.batches] == [4, 4, 2]
    seen = np.concatenate(recorder.batches)
    assert sorted(seen.tolist()) == list(range(10))


def test_each_epoch_reshuffles() -> None:
    """A fresh permutation per epoch, or every epoch sees the same batches."""
    recorder = Recorder(20)
    SGD(batch_size=5, lr=0.1, n_epochs=6,
        rng=np.random.default_rng(1)).minimize(recorder, np.ones(2))

    first_batches = [tuple(b.tolist()) for b in recorder.batches[:4]]
    later_batches = [tuple(b.tolist()) for b in recorder.batches[4:]]
    assert any(b not in first_batches for b in later_batches)


def test_observers_see_one_event_per_epoch() -> None:
    """The epoch is the fair unit of work, so it is the unit the history records."""
    X, y, _ = _dataset()
    history = History()
    SGD(batch_size=7, lr=0.05, n_epochs=9, rng=np.random.default_rng(0),
        observers=[history]).minimize(linear_regression(X, y), np.zeros(3))

    assert len(history.events) == 9
    assert [e.iteration for e in history.events] == list(range(1, 10))


def test_the_recorded_value_is_the_full_loss_not_a_batch_loss() -> None:
    X, y, _ = _dataset()
    loss = linear_regression(X, y)
    history = History()
    result = SGD(batch_size=7, lr=0.05, n_epochs=4, rng=np.random.default_rng(0),
                 observers=[history]).minimize(loss, np.zeros(3))

    assert history.values[-1] == pytest.approx(loss.value(result.x))
    assert history.events[-1].grad_norm == pytest.approx(
        float(np.linalg.norm(loss.gradient(result.x)))
    )


def test_the_step_size_follows_the_decay_schedule() -> None:
    """α_k = α₀/(1 + γk), indexed by epoch — the schedule of Lecture 3."""
    X, y, _ = _dataset()
    history = History()
    SGD(batch_size=10, lr=0.4, n_epochs=5, lr_decay=0.25, rng=np.random.default_rng(0),
        observers=[history]).minimize(linear_regression(X, y), np.zeros(3))

    expected = [0.4 / (1.0 + 0.25 * k) for k in range(5)]
    assert [e.step_size for e in history.events] == pytest.approx(expected)


def test_no_decay_by_default() -> None:
    X, y, _ = _dataset()
    history = History()
    SGD(batch_size=10, lr=0.4, n_epochs=4, rng=np.random.default_rng(0),
        observers=[history]).minimize(linear_regression(X, y), np.zeros(3))

    assert [e.step_size for e in history.events] == pytest.approx([0.4] * 4)


def test_decay_lowers_the_noise_floor() -> None:
    """The plateau of a constant step is set by the step, and shrinking it escapes.

    A weaker version of notebook 3 §3, small enough to run in a test: the decayed run
    must end well below the constant one, not merely below it.
    """
    X, _, y = _dataset(n=200, d=4, seed=3)
    loss = logistic_regression(X, y)
    w0 = np.zeros(4)

    constant = SGD(batch_size=8, lr=0.4, n_epochs=150,
                   rng=np.random.default_rng(7)).minimize(loss, w0)
    decayed = SGD(batch_size=8, lr=0.4, n_epochs=150, lr_decay=0.2,
                  rng=np.random.default_rng(7)).minimize(loss, w0)

    assert decayed.grad_norm < 0.1 * constant.grad_norm


def test_momentum_defaults_to_none_and_changes_the_run_when_set() -> None:
    X, y, _ = _dataset()
    loss = linear_regression(X, y)
    w0 = np.zeros(3)
    plain = SGD(batch_size=8, lr=0.05, n_epochs=5,
                rng=np.random.default_rng(2)).minimize(loss, w0).x
    heavy = SGD(batch_size=8, lr=0.05, n_epochs=5, momentum=0.9,
                rng=np.random.default_rng(2)).minimize(loss, w0).x
    assert not np.allclose(plain, heavy)


def test_sgd_reports_the_budget_it_spent() -> None:
    """Spending an epoch budget is not convergence — `MaxIterations` is not either."""
    X, y, _ = _dataset()
    result = SGD(batch_size=8, lr=0.05, n_epochs=7,
                 rng=np.random.default_rng(0)).minimize(linear_regression(X, y), np.zeros(3))
    assert result.iterations == 7
    assert result.converged is False
    assert result.x.shape == (3,)


def test_zero_epochs_returns_the_starting_point() -> None:
    X, y, _ = _dataset()
    loss = linear_regression(X, y)
    w0 = np.array([0.5, -0.5, 0.25])
    result = SGD(n_epochs=0, rng=np.random.default_rng(0)).minimize(loss, w0)
    assert result.x == pytest.approx(w0)
    assert result.value == pytest.approx(loss.value(w0))


def test_sgd_does_not_modify_the_starting_point() -> None:
    X, y, _ = _dataset()
    w0 = np.zeros(3)
    SGD(batch_size=8, lr=0.1, n_epochs=3,
        rng=np.random.default_rng(0)).minimize(linear_regression(X, y), w0)
    assert np.array_equal(w0, np.zeros(3))


def test_sgd_refuses_an_objective_that_cannot_be_sampled() -> None:
    """`Rosenbrock` is an `IObjective` but not a finite sum, so there is nothing to batch."""
    with pytest.raises(TypeError):
        SGD(rng=np.random.default_rng(0)).minimize(Rosenbrock(), np.array([-1.2, 1.0]))


# --------------------------------------------------------------------------- #
# Exercise 3 — Adam
# --------------------------------------------------------------------------- #


def test_the_first_adam_step_is_lr_times_the_sign_of_the_gradient() -> None:
    """At t = 1, m̂ = g and √v̂ = |g|, so the ratio is sign(g) — exactly.

    Checked over six orders of magnitude of gradient: that the size of the step does not
    depend on the size of the gradient is the whole of what Adam does.

    The tolerance is 10⁻⁴ rather than machine precision because of `eps`. It is added to
    √v̂ to keep the division safe, so it also costs a relative eps/|g| — a millionth at
    |g| = 10⁻³ here, and the reason `eps` must stay far below the gradients you expect.
    """
    for magnitude in (1e-3, 1.0, 1e3):
        # X = I and y = (2m, −2m) put the gradient at exactly (−m, +m) for w = 0.
        X = np.eye(2)
        y = np.array([2.0 * magnitude, -2.0 * magnitude])
        loss = GLMLoss(X, y, SquaredError())
        w0 = np.zeros(2)
        g = loss.gradient(w0)
        assert np.abs(g) == pytest.approx([magnitude, magnitude])

        result = Adam(batch_size=2, lr=0.1, n_epochs=1,
                      rng=np.random.default_rng(0)).minimize(loss, w0)

        assert result.x == pytest.approx(-0.1 * np.sign(g), rel=1e-4)


def test_the_first_step_is_bias_corrected() -> None:
    """Without the correction the first step would be (1−β₁)/√(1−β₂) ≈ 3.162 times larger.

    The usual "the averages start at zero so early steps are too small" is only half the
    story: both moments are biased towards zero, the step is their ratio, and here the
    uncorrected step comes out *bigger*.
    """
    loss = ConstantPerSample(4)
    w0 = np.array([1.0])
    result = Adam(batch_size=4, lr=0.1, n_epochs=1,
                  rng=np.random.default_rng(0)).minimize(loss, w0)

    moved = abs(float(result.x[0] - w0[0]))
    uncorrected = 0.1 * (1.0 - 0.9) / np.sqrt(1.0 - 0.999)
    assert moved == pytest.approx(0.1, rel=1e-6)
    assert moved != pytest.approx(uncorrected, rel=1e-3)


def test_bias_correction_counts_steps_not_epochs() -> None:
    """Two batches in one epoch are two updates, so t goes 1 then 2.

    Every row of this problem is identical, so the batch gradient is the same whichever
    half the shuffle produced — which makes the arithmetic below exact without pinning
    down the permutation. Indexing t by epoch leaves the second step at t = 1 and lands
    somewhere measurably different.
    """
    loss = ConstantPerSample(4)      # f(w) = ½w², so ∇f = w on every batch
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8

    w = np.array([1.0])
    m = np.zeros(1)
    v = np.zeros(1)
    for t in (1, 2):
        g = w.copy()                 # the gradient of this problem is w itself
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g**2
        w = w - lr * (m / (1.0 - b1**t)) / (np.sqrt(v / (1.0 - b2**t)) + eps)

    result = Adam(batch_size=2, lr=lr, n_epochs=1,
                  rng=np.random.default_rng(0)).minimize(loss, np.array([1.0]))
    assert result.x == pytest.approx(w, rel=1e-9)


def test_adam_observers_see_one_event_per_epoch() -> None:
    X, y, _ = _dataset()
    history = History()
    Adam(batch_size=7, lr=0.05, n_epochs=6, rng=np.random.default_rng(0),
         observers=[history]).minimize(linear_regression(X, y), np.zeros(3))
    assert [e.iteration for e in history.events] == list(range(1, 7))


def test_adam_ignores_the_global_numpy_seed() -> None:
    X, _, y = _dataset()
    loss = logistic_regression(X, y)
    w0 = np.zeros(3)
    np.random.seed(0)
    first = Adam(batch_size=8, lr=0.05, n_epochs=4,
                 rng=np.random.default_rng(5)).minimize(loss, w0).x
    np.random.seed(999)
    second = Adam(batch_size=8, lr=0.05, n_epochs=4,
                  rng=np.random.default_rng(5)).minimize(loss, w0).x
    assert np.array_equal(first, second)


def test_adam_refuses_an_objective_that_cannot_be_sampled() -> None:
    with pytest.raises(TypeError):
        Adam(rng=np.random.default_rng(0)).minimize(Rosenbrock(), np.array([-1.2, 1.0]))


def test_adam_makes_progress_on_a_well_scaled_problem() -> None:
    """A sanity run. Note the modest `lr`: Adam has a noise floor like any stochastic
    method, and at `lr = 0.1` this run stops at ‖∇f‖ ≈ 3·10⁻², eight times worse.
    "Use Adam and stop tuning" is not a thing.
    """
    X, y, _ = _dataset(n=120, d=3, seed=8)
    loss = linear_regression(X, y)
    w0 = np.zeros(3)
    result = Adam(batch_size=16, lr=0.02, n_epochs=60,
                  rng=np.random.default_rng(0)).minimize(loss, w0)
    assert result.value < loss.value(w0)
    assert result.grad_norm < 1e-2


def test_adam_beats_sgd_when_the_features_are_badly_scaled() -> None:
    """The lab's headline comparison, shrunk to test size.

    One column of order 1, one of order 10⁻³. SGD's largest stable step is set by the
    stiff direction and all the missing progress is in the soft one; there is no single
    number that serves both. Adam divides each coordinate by its own gradient scale and
    the problem goes away.
    """
    rng = np.random.default_rng(4242)
    n = 1500
    X = np.column_stack([rng.normal(size=n), rng.normal(scale=1e-3, size=n)])
    y = X @ np.array([2.0, -1500.0]) + rng.normal(scale=0.1, size=n)
    loss = GLMLoss(X, y, SquaredError())
    w0 = np.zeros(2)

    hessian = X.T @ X / n
    f_star = loss.value(np.linalg.solve(hessian, X.T @ y / n))

    # SGD gets three attempts spanning a factor of almost four in step size, and the
    # best of them counts. It is not being handicapped: raising the step does not help,
    # because the largest stable one is set by the stiff direction.
    sgd_gap = min(
        SGD(batch_size=32, lr=lr, n_epochs=60,
            rng=np.random.default_rng(1)).minimize(loss, w0).value - f_star
        for lr in (0.5, 1.0, 1.8)
    )
    adam_gap = Adam(batch_size=32, lr=2.0, n_epochs=60,
                    rng=np.random.default_rng(1)).minimize(loss, w0).value - f_star

    assert adam_gap < sgd_gap / 100.0
