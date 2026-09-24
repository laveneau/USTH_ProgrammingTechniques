"""[DAY 6] ElasticNet, the proximal gradient method, Huber and Poisson.

The soft-threshold itself is checked by hand in `test_day6_prox.py`; this file is
Exercises 1 and 2 as a whole.

Three of these tests are about design rather than arithmetic.
`test_proximal_gradient_never_asks_the_penalty_for_a_gradient` is the reason `L1.gradient`
is allowed to raise at all; `test_lasso_zeroes_exactly_while_ridge_only_shrinks` is the
one claim the entire day is built on; and `test_huber_with_a_huge_delta_is_squared_error`
asserts *bit* equality, not approximate equality, because for data whose residuals all sit
inside the quadratic branch the two losses are the same expression.
"""

import numpy as np
import pytest

from optlab.interfaces import IRegularizer
from optlab.losses import Huber, PoissonNLL, SquaredError
from optlab.objective_ops import RegularizedObjective
from optlab.observers import History
from optlab.optimizers import ProximalGradient, newton
from optlab.problems import GLMLoss
from optlab.regularizers import L1, L2, ElasticNet, NoRegularizer
from optlab.types import Vec

pytestmark = pytest.mark.day6

TRUE_SUPPORT = [0, 3, 7]
TRUE_VALUES = [3.0, -2.0, 1.5]


def _sparse_problem(n: int = 60, d: int = 20, seed: int = 7) -> tuple[GLMLoss, float]:
    """A design where only 3 of d coefficients are real, plus the largest safe step 1/L."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[TRUE_SUPPORT] = TRUE_VALUES
    y = X @ w_true + rng.normal(scale=0.3, size=n)
    smooth = GLMLoss(X, y, SquaredError())
    step = 1.0 / float(np.linalg.eigvalsh(X.T @ X / n)[-1])
    return smooth, step


def _prox_objective(w: Vec, v: Vec, t: float, reg: IRegularizer) -> float:
    return 0.5 * float(np.sum((w - v) ** 2)) + t * reg.value(w)


# --------------------------------------------------------------------------- #
# Exercise 1 — the penalties
# --------------------------------------------------------------------------- #


def test_l1_value_is_the_scaled_one_norm() -> None:
    assert L1(lam=0.5).value(np.array([3.0, -4.0, 0.0])) == pytest.approx(3.5)


def test_elastic_net_value_mixes_the_two_norms() -> None:
    w = np.array([3.0, -4.0])
    reg = ElasticNet(lam=2.0, alpha=0.25)
    expected = 2.0 * (0.25 * 7.0 + 0.5 * 0.75 * 25.0)
    assert reg.value(w) == pytest.approx(expected)


def test_elastic_net_at_the_two_extremes_is_l1_and_l2() -> None:
    """alpha = 1 must *be* lasso and alpha = 0 must *be* ridge — value and prox both.

    A prox that composed the two operators in the wrong order, or that forgot to scale
    the threshold by alpha, still passes the convexity contract test and fails here.
    """
    w = np.array([3.0, -0.4, 0.1, 0.0])
    t = 0.7
    lam = 0.5

    lasso_like = ElasticNet(lam=lam, alpha=1.0)
    assert lasso_like.value(w) == pytest.approx(L1(lam).value(w))
    np.testing.assert_allclose(lasso_like.prox(w, t), L1(lam).prox(w, t))

    ridge_like = ElasticNet(lam=lam, alpha=0.0)
    assert ridge_like.value(w) == pytest.approx(L2(lam).value(w))
    np.testing.assert_allclose(ridge_like.prox(w, t), L2(lam).prox(w, t))


def test_elastic_net_prox_thresholds_then_shrinks() -> None:
    """Both effects must be visible: an exact zero *and* a shrunk survivor.

    Soft-threshold alone would leave the survivor at |v| − tλα; shrink alone would
    produce no zero at all.
    """
    reg = ElasticNet(lam=1.0, alpha=0.5)
    out = reg.prox(np.array([3.0, 0.2]), 1.0)
    assert out[1] == 0.0
    assert out[0] == pytest.approx(2.5 / 1.5)


def test_every_prox_beats_its_neighbours() -> None:
    """The defining identity, on a coarse grid rather than at random.

    prox_{t·r}(v) = argmin_w ½‖w − v‖² + t·r(w), so nothing nearby may score better.
    """
    v = np.array([1.7, -0.3, 0.05, 0.0])
    for reg in (NoRegularizer(), L2(0.7), L1(0.5), ElasticNet(0.5, 0.6)):
        w = reg.prox(v, 0.5)
        best = _prox_objective(w, v, 0.5, reg)
        for k in range(v.size):
            for delta in (-0.2, -0.05, -1e-3, 1e-3, 0.05, 0.2):
                candidate = w.copy()
                candidate[k] += delta
                assert _prox_objective(candidate, v, 0.5, reg) >= best - 1e-12


# --------------------------------------------------------------------------- #
# Exercise 1 — ISTA and FISTA
# --------------------------------------------------------------------------- #


def test_proximal_gradient_ignores_its_objective_argument() -> None:
    """The smooth part and the penalty arrive in the constructor, so `minimize(None, x0)`
    is the documented call — and `IOptimizer` still holds, which is the point."""
    smooth, step = _sparse_problem()
    result = ProximalGradient(smooth, L1(0.1), step=step, max_iter=50).minimize(None, np.zeros(20))
    assert result.x.shape == (20,)


def test_proximal_gradient_never_asks_the_penalty_for_a_gradient() -> None:
    """The whole reason `L1.gradient` is allowed to raise.

    This regularizer raises something the test can recognise, so a single stray call is
    caught here rather than surfacing as a mysterious `NotImplementedError` from inside
    a student's own run.
    """

    class Tripwire(L1):
        def gradient(self, w: Vec) -> Vec:
            raise AssertionError("ProximalGradient must never call reg.gradient")

    smooth, step = _sparse_problem()
    ProximalGradient(smooth, Tripwire(0.1), step=step, max_iter=30).minimize(None, np.zeros(20))


def test_lasso_recovers_the_true_support() -> None:
    smooth, step = _sparse_problem()
    result = ProximalGradient(smooth, L1(0.1), step=step, max_iter=20000, tol=1e-12,
                              accelerated=True).minimize(None, np.zeros(20))
    assert result.converged is True
    np.testing.assert_array_equal(np.flatnonzero(result.x != 0.0), TRUE_SUPPORT)
    np.testing.assert_allclose(result.x[TRUE_SUPPORT], TRUE_VALUES, atol=0.2)


def test_lasso_zeroes_exactly_while_ridge_only_shrinks() -> None:
    """The claim the whole day rests on, and the reason `==` is the right comparison.

    Same optimizer, same data, same step, same tolerance. The *only* thing that changes
    is which `IRegularizer` object is passed in — which is the design point of day 4 and
    day 6 together.
    """
    smooth, step = _sparse_problem()
    settings = dict(step=step, max_iter=20000, tol=1e-12, accelerated=True)

    lasso = ProximalGradient(smooth, L1(0.1), **settings).minimize(None, np.zeros(20))
    ridge = ProximalGradient(smooth, L2(0.1), **settings).minimize(None, np.zeros(20))

    assert int(np.sum(lasso.x == 0.0)) == 17
    assert int(np.sum(ridge.x == 0.0)) == 0
    # Not "small": ridge's smallest coefficient is nowhere near the float 0.0.
    assert np.min(np.abs(ridge.x)) > 1e-4


def test_raising_lambda_far_enough_zeroes_everything() -> None:
    """The end of the regularization path: past λ ≥ max|Xᵀy|/n nothing survives."""
    smooth, step = _sparse_problem()
    result = ProximalGradient(smooth, L1(100.0), step=step, max_iter=500, tol=1e-12).minimize(
        None, np.zeros(20)
    )
    np.testing.assert_array_equal(result.x, np.zeros(20))


def test_ista_and_fista_agree_on_the_first_iteration() -> None:
    """FISTA's extrapolation weight is (t₀ − 1)/t₁ = 0 at the start, so step 1 is shared.

    A momentum term that was already active on the first step would mean the state was
    initialised wrongly, and the two curves would part company immediately.
    """
    smooth, step = _sparse_problem()
    runs = []
    for accelerated in (False, True):
        history = History()
        ProximalGradient(smooth, L1(0.1), step=step, max_iter=1, tol=-1.0,
                         accelerated=accelerated, observers=[history]).minimize(None, np.zeros(20))
        runs.append(history.events[0].x)
    np.testing.assert_array_equal(runs[0], runs[1])


def test_fista_finds_the_support_sooner_than_ista() -> None:
    """The real mechanism behind FISTA's advantage, which is not the O(1/k²) bound.

    While the support is wrong the method is searching a 20-dimensional problem; once it
    is right the problem has collapsed to 3 dimensions and is strongly convex there.
    Whoever identifies the active set first wins, and momentum does it sooner.
    """
    smooth, step = _sparse_problem()
    settled = {}
    for accelerated, name in ((False, "ista"), (True, "fista")):
        history = History()
        ProximalGradient(smooth, L1(0.1), step=step, max_iter=400, tol=-1.0,
                         accelerated=accelerated, observers=[history]).minimize(None, np.zeros(20))
        nnz = np.array([int(np.sum(e.x != 0.0)) for e in history.events])
        settled[name] = int(np.argmax(nnz == nnz[-1])) + 1
    assert settled["fista"] < settled["ista"]


def test_both_reach_the_same_minimum() -> None:
    """Acceleration changes the route, never the destination."""
    smooth, step = _sparse_problem()
    settings = dict(step=step, max_iter=50000, tol=1e-13)
    start = np.zeros(20)
    ista = ProximalGradient(smooth, L1(0.1), accelerated=False, **settings).minimize(None, start)
    fista = ProximalGradient(smooth, L1(0.1), accelerated=True, **settings).minimize(None, start)
    np.testing.assert_allclose(ista.x, fista.x, atol=1e-8)
    assert ista.value == pytest.approx(fista.value, abs=1e-12)


def test_the_recorded_value_is_the_full_objective() -> None:
    """f + r, not f alone.

    Reporting only the smooth half makes the penalty invisible in every convergence plot,
    and notebook 6 §2 subtracts these values from F* directly.
    """
    smooth, step = _sparse_problem()
    history = History()
    ProximalGradient(smooth, L1(0.1), step=step, max_iter=12, tol=-1.0,
                     observers=[history]).minimize(None, np.zeros(20))
    reg = L1(0.1)
    for event in history.events:
        assert event.value == pytest.approx(smooth.value(event.x) + reg.value(event.x))


def test_one_event_per_iteration_and_a_monotone_budget() -> None:
    smooth, step = _sparse_problem()
    history = History()
    result = ProximalGradient(smooth, L1(0.1), step=step, max_iter=25, tol=-1.0,
                              observers=[history]).minimize(None, np.zeros(20))
    assert result.iterations == 25
    assert result.converged is False
    assert [e.iteration for e in history.events] == list(range(1, 26))


def test_a_negative_tolerance_disables_the_early_return() -> None:
    """Notebook 6 uses `tol=-1.0` to force a fixed-length run; the residual is never < 0."""
    smooth, step = _sparse_problem()
    result = ProximalGradient(smooth, L1(0.1), step=step, max_iter=5000, tol=-1.0,
                              accelerated=True).minimize(None, np.zeros(20))
    assert result.iterations == 5000
    assert result.converged is False


def test_the_reported_residual_is_not_a_gradient_norm() -> None:
    """At the solution ‖w⁺ − w‖/α is 0, while ∇F does not exist at all.

    A student who stopped on ‖∇f(w)‖ of the smooth part alone would never converge: that
    quantity is λ at every zeroed coordinate, not 0.
    """
    smooth, step = _sparse_problem()
    lam = 0.1
    result = ProximalGradient(smooth, L1(lam), step=step, max_iter=20000, tol=1e-12,
                              accelerated=True).minimize(None, np.zeros(20))
    assert result.grad_norm <= 1e-12
    smooth_gradient = smooth.gradient(result.x)
    zeroed = result.x == 0.0
    assert np.max(np.abs(smooth_gradient[zeroed])) > 1e-3


def test_lasso_matches_sklearn() -> None:
    """The oracle. sklearn's `Lasso` minimises ‖y − Xw‖²/(2n) + α‖w‖₁, which is exactly
    `GLMLoss(X, y, SquaredError())` plus `L1(α)` — no rescaling needed."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    smooth, step = _sparse_problem()
    lam = 0.1
    mine = ProximalGradient(smooth, L1(lam), step=step, max_iter=200000, tol=1e-14,
                            accelerated=True).minimize(None, np.zeros(20))
    reference = linear_model.Lasso(
        alpha=lam, fit_intercept=False, tol=1e-14, max_iter=200000
    ).fit(smooth.X, smooth.y)
    np.testing.assert_allclose(mine.x, reference.coef_, atol=1e-10)


# --------------------------------------------------------------------------- #
# Exercise 2 — Huber and Poisson
# --------------------------------------------------------------------------- #


def test_huber_by_hand_on_both_sides_of_the_kink() -> None:
    """delta = 1: inside, ½r²; outside, |r| − ½. At r = 2 that is 1.5, not 2."""
    loss = Huber(delta=1.0)
    z = np.array([0.5, 2.0, -3.0])
    y = np.zeros(3)
    np.testing.assert_allclose(loss.value(z, y), [0.125, 1.5, 2.5])
    np.testing.assert_allclose(loss.d1(z, y), [0.5, 1.0, -1.0])
    np.testing.assert_allclose(loss.d2(z, y), [1.0, 0.0, 0.0])


def test_huber_is_continuously_differentiable_at_the_kink() -> None:
    """C¹ is the property that makes Huber usable by every optimizer of the week.

    Value and first derivative match from both sides at r = ±delta; only the *second*
    derivative jumps, which is why Newton still works but a finite-difference check of
    d2 must not be taken across the kink.
    """
    loss = Huber(delta=1.0)
    eps = 1e-9
    for side in (1.0, -1.0):
        z = np.array([side * (1.0 - eps), side * (1.0 + eps)])
        y = np.zeros(2)
        values = loss.value(z, y)
        slopes = loss.d1(z, y)
        assert values[0] == pytest.approx(values[1], abs=1e-8)
        assert slopes[0] == pytest.approx(slopes[1], abs=1e-8)


def test_huber_bounds_the_influence_of_any_observation() -> None:
    """|φ'| ≤ delta however far away the point is — that is the definition of robust."""
    loss = Huber(delta=1.5)
    z = np.array([1e3, -1e6, 0.0])
    y = np.zeros(3)
    assert np.max(np.abs(loss.d1(z, y))) <= 1.5 + 1e-12


def test_huber_with_a_huge_delta_is_squared_error() -> None:
    """Not "approaches": *is*, for data whose residuals all stay inside the quadratic
    branch. Same expression, so the fitted parameters agree in every bit."""
    m = 120
    t = np.linspace(-2.0, 2.0, m)
    X = np.column_stack([np.ones(m), t])
    y = X @ np.array([1.0, 2.0]) + np.random.default_rng(31).normal(scale=0.2, size=m)

    start = np.zeros(2)
    by_squared = newton(tol=1e-10, max_iter=300).minimize(GLMLoss(X, y, SquaredError()), start)
    by_huber = newton(tol=1e-10, max_iter=300).minimize(GLMLoss(X, y, Huber(100.0)), start)

    assert np.max(np.abs(X @ by_squared.x - y)) < 100.0      # nothing leaves the branch
    np.testing.assert_array_equal(by_huber.x, by_squared.x)


def test_huber_survives_outliers_that_destroy_least_squares() -> None:
    """20 corrupted points out of 120, all on the right: the slope, not just the fit.

    Squared error returns a slope of −0.065 where the truth is 2.0 — not degraded,
    reversed. Huber returns 1.70.
    """
    m = 120
    t = np.linspace(-2.0, 2.0, m)
    X = np.column_stack([np.ones(m), t])
    y = X @ np.array([1.0, 2.0]) + np.random.default_rng(31).normal(scale=0.2, size=m)
    corrupt = np.random.default_rng(120).choice(np.flatnonzero(t > 0.7), 20, replace=False)
    y[corrupt] -= 12.0

    start = np.zeros(2)
    by_squared = newton(tol=1e-10, max_iter=300).minimize(GLMLoss(X, y, SquaredError()), start)
    by_huber = newton(tol=1e-10, max_iter=600).minimize(GLMLoss(X, y, Huber(1.0)), start)

    assert by_squared.x[1] < 0.5          # the sign of the effect has been lost
    assert by_huber.x[1] > 1.5            # the story is still right


def test_poisson_by_hand() -> None:
    """φ(z, y) = e^z − y·z, so at z = 0 the value is 1 − 0 and the slope is 1 − y."""
    loss = PoissonNLL()
    z = np.zeros(3)
    y = np.array([0.0, 1.0, 4.0])
    np.testing.assert_allclose(loss.value(z, y), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(loss.d1(z, y), [1.0, 0.0, -3.0])
    np.testing.assert_allclose(loss.d2(z, y), [1.0, 1.0, 1.0])


def test_poisson_curvature_is_strictly_positive() -> None:
    """e^z > 0 everywhere, so XᵀDX/n is positive definite and Newton needs no damping."""
    z = np.linspace(-5.0, 5.0, 41)
    assert np.all(PoissonNLL().d2(z, np.zeros_like(z)) > 0.0)


def test_a_new_loss_needs_no_new_optimizer() -> None:
    """The open/closed claim the whole week has been building towards, as a test.

    `PoissonNLL` is one new `IPointwiseLoss`. Day 2's gradient descent, day 4's Newton
    and day 4's ridge all fit it with nothing else changed anywhere.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4)) * 0.5
    y = rng.poisson(np.exp(X @ np.array([0.3, -0.2, 0.5, 0.1]))).astype(float)
    loss = GLMLoss(X, y, PoissonNLL())

    plain = newton(tol=1e-12, max_iter=100).minimize(loss, np.zeros(4))
    penalised = newton(tol=1e-12, max_iter=100).minimize(
        RegularizedObjective(loss, L2(1.0)), np.zeros(4)
    )

    assert plain.converged is True and penalised.converged is True
    # Ridge pulls every coefficient towards zero, and must not merely reproduce the fit.
    assert np.linalg.norm(penalised.x) < np.linalg.norm(plain.x)


def test_poisson_regression_matches_sklearn() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4)) * 0.5
    y = rng.poisson(np.exp(X @ np.array([0.3, -0.2, 0.5, 0.1]))).astype(float)

    mine = newton(tol=1e-12, max_iter=100).minimize(GLMLoss(X, y, PoissonNLL()), np.zeros(4))
    reference = linear_model.PoissonRegressor(
        alpha=0.0, fit_intercept=False, tol=1e-12, max_iter=1000
    ).fit(X, y)
    np.testing.assert_allclose(mine.x, reference.coef_, atol=1e-7)
