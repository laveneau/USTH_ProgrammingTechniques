"""[DAY 4] Hessians, Newton, damping, and the regularizer adapter.

Read these as the specification of Exercises 2, 3 and 4. Exercise 1 has its own file,
`test_day4_cholesky.py`, and the solver contract lives under `tests/contracts/`.

Two of these check a *design* decision rather than a number.
`test_newton_lets_the_cholesky_failure_through` pins the propagate-do-not-swallow rule
that the whole of §3 of notebook 4 depends on, and
`test_a_line_search_cannot_repair_an_indefinite_hessian` pins the distinction the lecture
spends a page on: a line search fixes step *length* and can only veto a direction.
"""

import numpy as np
import pytest

from optlab.errors import NotPositiveDefiniteError
from optlab.interfaces import IObjective, ITwiceDifferentiable
from optlab.linalg import CholeskySolver
from optlab.linesearch import Armijo, FixedStep
from optlab.losses import LogisticNLL, SquaredError
from optlab.numerics import check_hessian
from optlab.objective_ops import RegularizedObjective
from optlab.observers import History
from optlab.optimizers import (
    DescentOptimizer,
    ModifiedNewton,
    NewtonDirection,
    SteepestDescent,
    newton,
)
from optlab.problems import GLMLoss, Quadratic, Rosenbrock, linear_regression
from optlab.regularizers import L2, NoRegularizer
from optlab.stopping import AnyOf, GradientNormBelow, MaxIterations
from optlab.types import Mat, Vec

pytestmark = pytest.mark.day4


class DoubleWell(IObjective, ITwiceDifferentiable):
    """f(x) = (x² − 1)², in one variable. Minima at ±1, a local MAXIMUM at 0.

    The example from Lecture 4: at x = 0.3 the curvature is −2.92, so the Newton
    direction points uphill and Cholesky refuses to produce it at all.
    """

    def value(self, x: Vec) -> float:
        return float((x[0] ** 2 - 1.0) ** 2)

    def gradient(self, x: Vec) -> Vec:
        return np.array([4.0 * x[0] * (x[0] ** 2 - 1.0)])

    def hessian(self, x: Vec) -> Mat:
        return np.array([[12.0 * x[0] ** 2 - 4.0]])


class OnlyFirstOrder(IObjective):
    """An objective with no Hessian at all — `ITwiceDifferentiable` is a separate mixin."""

    def value(self, x: Vec) -> float:
        return float(x @ x)

    def gradient(self, x: Vec) -> Vec:
        return 2.0 * x


def _design(n: int = 30, d: int = 4, seed: int = 0) -> tuple[Mat, Vec, Vec]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    return X, y, (rng.random(n) < 0.5).astype(float)


def _stop(tol: float = 1e-10, max_iter: int = 60) -> AnyOf:
    return AnyOf(GradientNormBelow(tol), MaxIterations(max_iter))


# --------------------------------------------------------------------------- #
# Exercise 2 — check_hessian and the three hessians
# --------------------------------------------------------------------------- #


def test_check_hessian_accepts_a_correct_hessian() -> None:
    """A Hessian *is* the Jacobian of the gradient, so day 1's sweep does the work."""
    problem = Rosenbrock()
    x = np.array([-1.2, 1.0, 0.7, 1.3])
    assert check_hessian(problem.gradient, problem.hessian, x, tol=1e-4) < 1e-4


def test_check_hessian_rejects_a_wrong_hessian() -> None:
    """A check that cannot fail is not a check."""
    problem = Rosenbrock()
    x = np.array([-1.2, 1.0])
    with pytest.raises(AssertionError):
        check_hessian(problem.gradient, lambda w: np.eye(w.size), x)


def test_quadratic_hessian_is_the_matrix_itself() -> None:
    """Constant — which is exactly why Newton finishes a quadratic in one step."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    problem = Quadratic(A, np.array([1.0, -2.0]))
    np.testing.assert_allclose(problem.hessian(np.zeros(2)), A)
    np.testing.assert_allclose(problem.hessian(np.array([7.0, -9.0])), A)


def test_rosenbrock_hessian_by_hand_at_the_minimum() -> None:
    """At (1, 1): H = [[802, −400], [−400, 200]], which is worth checking on paper."""
    H = Rosenbrock().hessian(np.array([1.0, 1.0]))
    np.testing.assert_allclose(H, np.array([[802.0, -400.0], [-400.0, 200.0]]))


def test_rosenbrock_hessian_is_indefinite_above_the_parabola() -> None:
    """det H = 400 − 80000y + 80000x², so H fails to be PD exactly where y > x² + 1/200."""
    assert np.linalg.eigvalsh(Rosenbrock().hessian(np.array([0.0, 2.0])))[0] < 0.0
    assert np.linalg.eigvalsh(Rosenbrock().hessian(np.array([0.0, -2.0])))[0] > 0.0


def test_glm_hessian_by_hand(hand_X: Mat, hand_y_linear: Vec) -> None:
    """Squared error has φ'' = 1, so H = XᵀX/n = (1·1 + 2·2)/2 = 2.5."""
    loss = linear_regression(hand_X, hand_y_linear)
    np.testing.assert_allclose(loss.hessian(np.zeros(1)), np.array([[2.5]]))


def test_glm_hessian_matches_finite_differences() -> None:
    X, y_linear, y_binary = _design()
    w = np.array([0.3, -0.2, 0.1, 0.4])
    for loss in (linear_regression(X, y_linear), GLMLoss(X, y_binary, LogisticNLL())):
        assert check_hessian(loss.gradient, loss.hessian, w, tol=1e-4) < 1e-4


def test_every_hessian_is_symmetric() -> None:
    X, y, _ = _design()
    w = np.array([0.3, -0.2, 0.1, 0.4])
    for problem, point in (
        (Quadratic.ill_conditioned(4, 100.0), np.array([1.0, -2.0, 0.5, 3.0])),
        (Rosenbrock(), np.array([-1.2, 1.0, 0.7, 1.3])),
        (linear_regression(X, y), w),
    ):
        H = problem.hessian(point)
        np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- #
# Exercise 2 — NewtonDirection and newton()
# --------------------------------------------------------------------------- #


def test_newton_solves_a_quadratic_in_one_iteration() -> None:
    """The model is not a model here, it is the function — so one step is exact."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, -2.0])
    result = newton().minimize(Quadratic(A, b), np.array([5.0, -7.0]))
    assert result.iterations == 1
    assert result.converged is True
    np.testing.assert_allclose(result.x, np.linalg.solve(A, b), rtol=1e-10, atol=1e-12)


def test_newton_on_linear_regression_is_the_normal_equations() -> None:
    """Least squares is a quadratic in w, so the same one step lands on the closed form."""
    X, y, _ = _design()
    result = newton().minimize(linear_regression(X, y), np.zeros(X.shape[1]))
    closed_form = np.linalg.solve(X.T @ X / X.shape[0], X.T @ y / X.shape[0])
    assert result.iterations == 1
    np.testing.assert_allclose(result.x, closed_form, rtol=1e-9, atol=1e-11)


def test_newton_and_gradient_descent_find_the_same_logistic_minimizer() -> None:
    """Different routes, one minimum — and note the iteration counts while you are here.

    Newton: 3 iterations. Steepest descent with the same line search and a *looser* target:
    128. That ratio is the whole argument for paying for a Hessian.
    """
    X, _, y = _design()
    loss = GLMLoss(X, y, LogisticNLL())
    w0 = np.zeros(X.shape[1])

    by_newton = newton().minimize(loss, w0)
    by_descent = DescentOptimizer(
        SteepestDescent(), Armijo(), _stop(tol=1e-8, max_iter=20000)
    ).minimize(loss, w0)

    assert by_descent.converged is True
    assert by_newton.iterations < 15
    assert by_newton.iterations < by_descent.iterations / 10
    np.testing.assert_allclose(by_newton.x, by_descent.x, rtol=1e-5, atol=1e-6)


def test_the_newton_error_squares_at_every_step() -> None:
    """e_{k+1}/e_k² stays bounded — the definition of quadratic convergence."""
    X, _, y = _design(n=200, d=5, seed=4)
    loss = GLMLoss(X, y, LogisticNLL())
    history = History()
    DescentOptimizer(
        NewtonDirection(CholeskySolver()), Armijo(), _stop(tol=1e-13), observers=[history]
    ).minimize(loss, np.zeros(5))

    w_star = history.events[-1].x
    errors = [float(np.linalg.norm(e.x - w_star)) for e in history.events]
    ratios = [
        errors[k + 1] / errors[k] ** 2
        for k in range(len(errors) - 1)
        if errors[k] > 1e-8          # above this the last steps only measure float64
    ]
    assert ratios, "the run was too short to see the rate"
    assert max(ratios) < 50.0


def test_newton_direction_solves_the_newton_equation() -> None:
    """H d = −g, and nothing else: no fallback, no scaling, no inverse."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    problem = Quadratic(A, np.array([1.0, -2.0]))
    x = np.array([0.5, 0.5])
    g = problem.gradient(x)
    d = NewtonDirection(CholeskySolver()).direction(problem, x, g)
    np.testing.assert_allclose(A @ d, -g, rtol=1e-10, atol=1e-12)


def test_newton_direction_needs_a_hessian() -> None:
    """`IDirectionRule` promises only an `IObjective`; a Hessian is a separate contract."""
    problem = OnlyFirstOrder()
    x = np.ones(3)
    with pytest.raises(TypeError):
        NewtonDirection(CholeskySolver()).direction(problem, x, problem.gradient(x))


# --------------------------------------------------------------------------- #
# Exercise 3 — the indefinite Hessian, and what does and does not repair it
# --------------------------------------------------------------------------- #


def test_newton_lets_the_cholesky_failure_through() -> None:
    """Propagate, do not fall back to −g.

    The exception says the quadratic model has a direction of negative curvature. Swapping
    in the gradient would hide that and leave the caller thinking Newton had run.
    """
    problem = DoubleWell()
    x = np.array([0.3])
    with pytest.raises(NotPositiveDefiniteError):
        NewtonDirection(CholeskySolver()).direction(problem, x, problem.gradient(x))


def test_a_line_search_cannot_repair_an_indefinite_hessian() -> None:
    """The distinction the lecture spends a page on, as an executable claim.

    Both line searches raise, and both raise the *same* error, because neither is ever
    reached: no direction was produced to take a step along. A line search chooses step
    length; this is a step direction problem.
    """
    problem = DoubleWell()
    x0 = np.array([0.3])
    messages = []
    for line_search in (FixedStep(1.0), Armijo()):
        with pytest.raises(NotPositiveDefiniteError) as caught:
            DescentOptimizer(
                NewtonDirection(CholeskySolver()), line_search, _stop()
            ).minimize(problem, x0)
        messages.append(str(caught.value))
    assert messages[0] == messages[1]


def test_modified_newton_reaches_the_minimum_of_the_double_well() -> None:
    """From x₀ = 0.3, where the textbook iteration converges to the maximum at x = 0."""
    history = History()
    result = DescentOptimizer(
        ModifiedNewton(CholeskySolver()), Armijo(), _stop(), observers=[history]
    ).minimize(DoubleWell(), np.array([0.3]))

    assert result.converged is True
    assert result.x[0] == pytest.approx(1.0, abs=1e-8)
    assert result.iterations < 15

    values = history.values
    assert all(values[k + 1] < values[k] for k in range(len(values) - 1)), (
        "the damped run must decrease f at every single step"
    )


def test_modified_newton_is_plain_newton_where_the_hessian_is_positive_definite() -> None:
    """The damping must get out of the way, or quadratic convergence is lost.

    On a positive definite problem the undamped solve succeeds and τ is never used, so
    the two directions agree to the last bit.
    """
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    problem = Quadratic(A, np.array([1.0, -2.0]))
    x = np.array([0.5, -0.25])
    g = problem.gradient(x)
    plain = NewtonDirection(CholeskySolver()).direction(problem, x, g)
    damped = ModifiedNewton(CholeskySolver()).direction(problem, x, g)
    np.testing.assert_allclose(damped, plain, rtol=1e-12, atol=1e-14)


def test_modified_newton_returns_a_descent_direction_on_an_indefinite_hessian() -> None:
    """gᵀd < 0 where the undamped Newton direction points uphill."""
    problem = DoubleWell()
    x = np.array([0.3])
    g = problem.gradient(x)
    d = ModifiedNewton(CholeskySolver()).direction(problem, x, g)
    assert float(g @ d) < 0.0


def test_the_damping_does_not_persist_between_calls() -> None:
    """τ restarts from `tau0` every call.

    A τ that only ever grew would turn the method into a very short steepest descent for
    the rest of the run. Asking for a hard point first must not degrade the easy one after.
    """
    rule = ModifiedNewton(CholeskySolver())
    problem = DoubleWell()
    rule.direction(problem, np.array([0.3]), problem.gradient(np.array([0.3])))

    easy = np.array([1.5])
    g = problem.gradient(easy)
    after = rule.direction(problem, easy, g)
    fresh = ModifiedNewton(CholeskySolver()).direction(problem, easy, g)
    np.testing.assert_allclose(after, fresh, rtol=1e-12, atol=1e-14)


def test_modified_newton_gives_up_rather_than_lying() -> None:
    """Past `tau_max` it raises, instead of returning something that is not a direction.

    At x = 0.3 the curvature is -2.92, so no tau below 2.92 can make H + tau*I positive
    definite. Capped at tau = 1 the loop must run out and re-raise. Falling back to -g
    here would silently turn Newton into steepest descent.
    """
    problem = DoubleWell()
    x = np.array([0.3])
    g = problem.gradient(x)
    with pytest.raises(NotPositiveDefiniteError):
        ModifiedNewton(CholeskySolver(), tau0=1e-3, tau_max=1.0).direction(problem, x, g)

    # With room to grow, the very same rule succeeds: it is the cap that bit, not a bug.
    generous = ModifiedNewton(CholeskySolver(), tau0=1e-3, tau_max=1e12)
    assert float(g @ generous.direction(problem, x, g)) < 0.0


# --------------------------------------------------------------------------- #
# Exercise 4 — ridge, for free
# --------------------------------------------------------------------------- #


def test_no_regularizer_is_the_neutral_element() -> None:
    w = np.array([3.0, -0.4, 0.1])
    reg = NoRegularizer()
    assert reg.value(w) == 0.0
    np.testing.assert_allclose(reg.gradient(w), np.zeros(3))
    np.testing.assert_allclose(reg.prox(w, 0.7), w)


def test_l2_value_gradient_and_prox() -> None:
    w = np.array([3.0, -4.0])
    reg = L2(lam=0.5)
    assert reg.value(w) == pytest.approx(0.5 * 0.5 * 25.0)
    np.testing.assert_allclose(reg.gradient(w), 0.5 * w)
    np.testing.assert_allclose(reg.prox(w, 2.0), w / (1.0 + 0.5 * 2.0))


def test_regularized_objective_adds_the_two_parts() -> None:
    X, y, _ = _design()
    inner = GLMLoss(X, y, SquaredError())
    reg = L2(lam=0.7)
    combined = RegularizedObjective(inner, reg)
    w = np.array([0.3, -0.2, 0.1, 0.4])

    assert combined.value(w) == pytest.approx(inner.value(w) + reg.value(w))
    np.testing.assert_allclose(combined.gradient(w), inner.gradient(w) + reg.gradient(w))


def test_the_regularized_gradient_is_the_gradient_of_the_regularized_value() -> None:
    """Checked the day-1 way, because an adapter is exactly where a sign slips."""
    from optlab.numerics import check_gradient

    X, y, _ = _design()
    combined = RegularizedObjective(GLMLoss(X, y, SquaredError()), L2(lam=0.7))
    assert check_gradient(combined.value, combined.gradient, np.array([0.3, -0.2, 0.1, 0.4])) < 1e-6


def test_ridge_lifts_every_eigenvalue_by_lambda() -> None:
    """Which is why ridge can never break Cholesky — and it is the last piece of day 5."""
    X, y, _ = _design()
    lam = 0.7
    inner = GLMLoss(X, y, SquaredError())
    combined = RegularizedObjective(inner, L2(lam))
    w = np.zeros(X.shape[1])

    before = np.linalg.eigvalsh(inner.hessian(w))
    after = np.linalg.eigvalsh(combined.hessian(w))
    np.testing.assert_allclose(after - before, np.full(X.shape[1], lam), rtol=1e-9, atol=1e-11)


def test_newton_on_a_ridge_objective_matches_the_closed_form() -> None:
    """The payoff: ridge regression with no new optimizer, no new loss, no new anything."""
    X, y, _ = _design(n=60, d=5, seed=1)
    n, d = X.shape
    lam = 0.7
    ridge = RegularizedObjective(GLMLoss(X, y, SquaredError()), L2(lam))

    result = newton().minimize(ridge, np.zeros(d))
    closed_form = np.linalg.solve(X.T @ X / n + lam * np.eye(d), X.T @ y / n)

    assert result.iterations == 1
    np.testing.assert_allclose(result.x, closed_form, rtol=1e-9, atol=1e-11)


def test_a_regularized_objective_without_a_hessian_says_so() -> None:
    """Only `hessian` needs the inner objective to be twice differentiable."""
    combined = RegularizedObjective(OnlyFirstOrder(), L2(0.5))
    w = np.ones(3)
    assert combined.value(w) == pytest.approx(3.0 + 0.5 * 0.5 * 3.0)
    with pytest.raises(TypeError):
        combined.hessian(w)
