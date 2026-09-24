"""[DAY 5] Residuals, Jacobians, Gauss-Newton and Levenberg-Marquardt.

Read these as the specification of the three exercises.

Two of them deserve a word before you start. `test_gauss_newton_diverges_from_the_hard_start`
is the test Exercise 2 asks you to write and **leave in**: it asserts a failure, on
purpose, because that failure is the reason Levenberg-Marquardt exists and the class is
unintelligible without it. And `test_levenberg_marquardt_converges_to_a_worse_minimum`
asserts that a run reporting `converged=True` found an answer 178 times worse than the
best one — `converged` means "the gradient is small *here*", never "this is the best".
"""

import numpy as np
import pytest

from optlab.errors import NotPositiveDefiniteError
from optlab.interfaces import ILeastSquaresProblem, ILinearSolver
from optlab.linalg import CholeskySolver, cholesky
from optlab.numerics import check_gradient, numerical_jacobian
from optlab.observers import History
from optlab.optimizers import GaussNewton, LevenbergMarquardt
from optlab.problems import Quadratic
from optlab.problems.curve_fitting import ExpDecay, GaussianPeak, Sinusoid
from optlab.types import Mat, Vec

pytestmark = pytest.mark.day5

EXP_TRUTH = np.array([2.5, 1.3, 0.5])


class CountingSolver(ILinearSolver):
    """A decorator over the real solver, to prove the injected one is the one used."""

    def __init__(self, inner: ILinearSolver) -> None:
        self.inner = inner
        self.calls = 0

    def solve(self, A: Mat, b: Vec) -> Vec:
        self.calls += 1
        return self.inner.solve(A, b)


def _exp_problem(noise: float = 0.0, m: int = 40, seed: int = 11) -> ExpDecay:
    t = np.linspace(0.0, 3.0, m)
    y = EXP_TRUTH[0] * np.exp(-EXP_TRUTH[1] * t) + EXP_TRUTH[2]
    if noise:
        y = y + np.random.default_rng(seed).normal(scale=noise, size=t.size)
    return ExpDecay(t, y)


def _sinusoid() -> Sinusoid:
    """The notebook's data: a = 2, omega = 3, phi = 0.7, sigma = 0.1."""
    t = np.linspace(0.0, 6.0, 120)
    y = 2.0 * np.sin(3.0 * t + 0.7) + np.random.default_rng(23).normal(scale=0.1, size=t.size)
    return Sinusoid(t, y)


def _cost(problem: ILeastSquaresProblem, x: Vec) -> float:
    r = problem.residuals(x)
    return 0.5 * float(r @ r)


def _grad(problem: ILeastSquaresProblem, x: Vec) -> Vec:
    return problem.jacobian(x).T @ problem.residuals(x)


# --------------------------------------------------------------------------- #
# Exercise 1 — residuals and Jacobians
# --------------------------------------------------------------------------- #


def test_exp_decay_residuals_by_hand() -> None:
    """At t = 0 the model is a + c whatever b is, which makes one row checkable on paper."""
    problem = ExpDecay(np.array([0.0, 0.0]), np.array([1.0, 4.0]))
    np.testing.assert_allclose(problem.residuals(np.array([2.5, 99.0, 0.5])), [2.0, -1.0])


def test_residuals_are_model_minus_data_not_the_other_way_round() -> None:
    """A flipped sign survives every symmetric check: ‖r‖² and JᵀJ are both blind to it.

    It shows up in `g = Jᵀr`, which then points uphill, and the optimizer walks away from
    the solution while every "is my Jacobian right" test stays green. Pin the orientation.
    """
    problem = ExpDecay(np.array([0.0]), np.array([1.0]))
    assert problem.residuals(np.array([3.0, 1.0, 0.0]))[0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("problem", "x"),
    [
        (ExpDecay(np.linspace(0.0, 3.0, 12), np.linspace(1.0, 0.2, 12)),
         np.array([2.0, 0.8, 0.3])),
        (GaussianPeak(np.linspace(-2.0, 4.0, 15), np.linspace(0.0, 1.0, 15)),
         np.array([3.0, 1.2, 0.8])),
        (Sinusoid(np.linspace(0.0, 6.0, 18), np.linspace(-1.0, 1.0, 18)),
         np.array([2.0, 3.0, 0.7])),
    ],
    ids=["ExpDecay", "GaussianPeak", "Sinusoid"],
)
def test_the_analytic_jacobian_matches_finite_differences(
    problem: ILeastSquaresProblem, x: Vec
) -> None:
    """Exercise 1's own instruction, as a test. A sign error here does not raise — it
    just makes the optimizer fail mysteriously, and you lose an hour."""
    numeric = numerical_jacobian(problem.residuals, x)
    analytic = problem.jacobian(x)
    assert analytic.shape == numeric.shape
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("problem", "x"),
    [
        (ExpDecay(np.linspace(0.0, 3.0, 12), np.linspace(1.0, 0.2, 12)),
         np.array([2.0, 0.8, 0.3])),
        (GaussianPeak(np.linspace(-2.0, 4.0, 15), np.linspace(0.0, 1.0, 15)),
         np.array([3.0, 1.2, 0.8])),
        (Sinusoid(np.linspace(0.0, 6.0, 18), np.linspace(-1.0, 1.0, 18)),
         np.array([2.0, 3.0, 0.7])),
    ],
    ids=["ExpDecay", "GaussianPeak", "Sinusoid"],
)
def test_jt_r_is_the_gradient_of_half_r_squared(problem: ILeastSquaresProblem, x: Vec) -> None:
    """Day 1's `check_gradient`, still earning its keep three labworks later."""
    assert check_gradient(lambda z: _cost(problem, z), lambda z: _grad(problem, z), x) < 1e-6


def test_the_jacobian_does_not_depend_on_the_data() -> None:
    """r = model(x) − y, so ∂r/∂x = ∂model/∂x and y drops out of the derivative entirely.

    Worth knowing, and worth testing: a Jacobian that changes with y has y somewhere it
    should not be.
    """
    t = np.linspace(0.0, 3.0, 9)
    x = np.array([2.0, 0.8, 0.3])
    np.testing.assert_allclose(
        ExpDecay(t, np.zeros(9)).jacobian(x), ExpDecay(t, np.ones(9) * 17.0).jacobian(x)
    )


def test_a_perfect_fit_has_zero_residuals() -> None:
    """The truth reproduces the data exactly on noiseless data — the sanity check that
    catches a model written with the wrong sign in the exponent."""
    problem = _exp_problem()
    np.testing.assert_allclose(problem.residuals(EXP_TRUTH), np.zeros(40), atol=1e-14)


# --------------------------------------------------------------------------- #
# Exercise 2 — Gauss-Newton
# --------------------------------------------------------------------------- #


def test_gauss_newton_needs_a_least_squares_problem() -> None:
    """A scalar `IObjective` does not expose r and J, and those are the whole method."""
    with pytest.raises(TypeError):
        GaussNewton(CholeskySolver()).minimize(Quadratic(np.eye(2), np.zeros(2)), np.zeros(2))


def test_gauss_newton_recovers_the_parameters_from_a_good_start() -> None:
    """(1, 1, 1) on noiseless data: a handful of iterations, and the truth to the last bits."""
    result = GaussNewton(CholeskySolver()).minimize(_exp_problem(), np.array([1.0, 1.0, 1.0]))
    assert result.converged is True
    assert result.iterations <= 10
    np.testing.assert_allclose(result.x, EXP_TRUTH, atol=1e-8)


def test_gauss_newton_solves_a_linear_problem_in_one_step() -> None:
    """When the model is linear in x, JᵀJ is the exact Hessian and GN *is* Newton.

    `ExpDecay` with b pinned is not linear, so the test builds a genuinely linear
    least-squares problem: r(x) = Ax − b. One step must land on the normal-equation
    solution, and the second must find nothing left to do.
    """

    class Linear(ILeastSquaresProblem):
        def __init__(self, A: Mat, b: Vec) -> None:
            self.A = A
            self.b = b

        def residuals(self, x: Vec) -> Vec:
            return self.A @ x - self.b

        def jacobian(self, x: Vec) -> Mat:
            return self.A

    rng = np.random.default_rng(5)
    A = rng.normal(size=(20, 3))
    b = rng.normal(size=20)
    problem = Linear(A, b)

    result = GaussNewton(CholeskySolver()).minimize(problem, np.array([9.0, -4.0, 2.0]))
    assert result.iterations == 1
    np.testing.assert_allclose(result.x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10)


def test_gauss_newton_uses_the_injected_solver() -> None:
    """Dependency injection, checked rather than assumed: no `np.linalg.solve` inside."""
    solver = CountingSolver(CholeskySolver())
    GaussNewton(solver).minimize(_exp_problem(), np.array([1.0, 1.0, 1.0]))
    assert solver.calls > 0


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_gauss_newton_diverges_from_the_hard_start() -> None:
    """**Exercise 2 asks you to write this and leave it in.**

    From (1, 10, 1) the model is nearly constant past t ≈ 0.5, so the columns of J for a
    and c are nearly parallel and JᵀJ is near-singular. The first step alone takes the
    cost from 6.6 to 9.3e+85. This is not a defect to be patched: it is the motivation
    for Levenberg-Marquardt, and a test that records it stops anyone "fixing" GN into
    something that hides it.
    """
    problem = _exp_problem()
    result = GaussNewton(CholeskySolver()).minimize(problem, np.array([1.0, 10.0, 1.0]))
    assert result.converged is False
    assert not np.allclose(result.x, EXP_TRUTH, atol=1e-3)


def test_the_first_gauss_newton_step_from_the_hard_start_is_catastrophic() -> None:
    """The mechanism behind the previous test, isolated from the loop.

    Nothing in Gauss-Newton limits the length of δ. Notebook 2 solved exactly this
    problem for gradient descent with a line search; LM solves it a different way.
    """
    problem = _exp_problem()
    x = np.array([1.0, 10.0, 1.0])
    J = problem.jacobian(x)
    delta = np.linalg.solve(J.T @ J, -J.T @ problem.residuals(x))
    assert _cost(problem, x) < 10.0
    assert _cost(problem, x + delta) > 1e20


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_gauss_newton_reports_which_failure_happened() -> None:
    """Two structurally different failures, and the `message` has to tell them apart.

    A near-singular JᵀJ is refused by the day-4 Cholesky ("normal equations failed"); an
    unbounded step walks the iterate off the edge of float64 ("iterates left the finite
    range"). Diagnosing the first as the second sends you looking in the wrong place.
    """
    t = np.linspace(0.0, 4.0, 60)
    y = 5.0 * np.exp(-1.3 * t) + 1.0 + np.random.default_rng(3).normal(scale=0.05, size=t.size)
    problem = ExpDecay(t, y)

    solver = GaussNewton(CholeskySolver(), max_iter=100)
    singular = solver.minimize(problem, np.array([1.0, 0.1, 0.0]))
    unbounded = solver.minimize(problem, np.array([0.5, 0.05, 4.0]))

    assert singular.converged is False and unbounded.converged is False
    assert "normal equations failed" in singular.message
    assert "finite range" in unbounded.message
    assert not np.all(np.isfinite(unbounded.x))


def test_gauss_newton_reports_one_event_per_step() -> None:
    result_history = History()
    result = GaussNewton(CholeskySolver(), observers=[result_history]).minimize(
        _exp_problem(), np.array([1.0, 1.0, 1.0])
    )
    assert len(result_history.events) == result.iterations
    assert [e.iteration for e in result_history.events] == list(range(1, result.iterations + 1))
    values = result_history.values
    assert values[-1] < values[0]


def test_gauss_newton_does_nothing_when_it_starts_at_the_solution() -> None:
    """The convergence test is at the top of the loop, so a free lunch costs no steps."""
    result = GaussNewton(CholeskySolver()).minimize(_exp_problem(), EXP_TRUTH.copy())
    assert result.converged is True
    assert result.iterations == 0


# --------------------------------------------------------------------------- #
# Exercise 3 — Levenberg-Marquardt
# --------------------------------------------------------------------------- #


def test_levenberg_marquardt_survives_the_start_that_killed_gauss_newton() -> None:
    """The headline claim of the exercise, and the reason the class exists."""
    problem = _exp_problem()
    hard = np.array([1.0, 10.0, 1.0])
    assert GaussNewton(CholeskySolver()).minimize(problem, hard).converged is False

    result = LevenbergMarquardt(CholeskySolver(), max_iter=300).minimize(problem, hard)
    assert result.converged is True
    np.testing.assert_allclose(result.x, EXP_TRUTH, atol=1e-8)


#: J with two identical columns. JᵀJ = [[14, 14], [14, 14]] in exact integer arithmetic,
#: so the second Cholesky pivot is exactly 0 — singular, not merely ill-conditioned, and
#: the same on every platform.
DEGENERATE_J = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


class RankDeficient(ILeastSquaresProblem):
    """r(x) = J·x − b with the two parameters genuinely indistinguishable.

    Only the sum x0 + x1 is determined; the cost is minimised at 17/14, along a whole
    line of equally good parameter vectors.
    """

    def __init__(self) -> None:
        self.b = np.array([1.0, 2.0, 4.0])

    def residuals(self, x: Vec) -> Vec:
        return DEGENERATE_J @ x - self.b

    def jacobian(self, x: Vec) -> Mat:
        return DEGENERATE_J


def test_damping_makes_any_normal_matrix_factorizable() -> None:
    """The guarantee, stated directly on the matrices: JᵀJ ⪰ 0, so JᵀJ + λI ≻ 0 for λ > 0.

    This is what makes LM's solve exception-free by construction, and it is why the
    reference implementation has no `except NotPositiveDefiniteError` there to catch
    anything with — an unreachable handler is a claim the reader cannot check.
    """
    normal = DEGENERATE_J.T @ DEGENERATE_J
    with pytest.raises(NotPositiveDefiniteError):
        cholesky(normal)
    for lam in (1e-12, 1e-6, 1e-3, 1.0, 1e6):
        cholesky(normal + lam * np.eye(2))          # must not raise, for any λ > 0


def test_cholesky_can_never_fail_inside_levenberg_marquardt() -> None:
    """The same guarantee through the optimizer: GN is refused, LM is not."""
    problem = RankDeficient()

    refused = GaussNewton(CholeskySolver(), max_iter=5).minimize(problem, np.zeros(2))
    assert refused.converged is False
    assert "normal equations failed" in refused.message

    result = LevenbergMarquardt(CholeskySolver(), max_iter=300).minimize(problem, np.zeros(2))
    assert float(result.x[0] + result.x[1]) == pytest.approx(17.0 / 14.0, abs=1e-6)


def test_levenberg_marquardt_rejects_a_step_that_makes_things_worse() -> None:
    """The gain ratio, observed from outside: on a rejection x does not move.

    From the notebook's hard start the first two attempts are rejected (ρ = −85 and
    −1.5), so the first two recorded events must sit at the same cost and report a step
    length of zero.
    """
    t = np.linspace(0.0, 4.0, 60)
    y = 5.0 * np.exp(-1.3 * t) + 1.0 + np.random.default_rng(3).normal(scale=0.05, size=t.size)
    history = History()
    LevenbergMarquardt(CholeskySolver(), max_iter=200, observers=[history]).minimize(
        ExpDecay(t, y), np.array([0.5, 0.05, 4.0])
    )

    rejected = [e for e in history.events if e.step_size == 0.0]
    assert rejected, "no step was rejected -- the gain ratio is not being consulted"
    assert history.values[0] == pytest.approx(history.values[1])


def test_lambda_falls_when_the_model_is_good_so_lm_becomes_gauss_newton() -> None:
    """From an easy start λ collapses and LM costs exactly what GN costs.

    If damping cost anything on easy problems nobody would use LM, so this equality of
    iteration counts is a real requirement and not a coincidence.
    """
    problem = _exp_problem(noise=0.02)
    start = np.array([2.0, 1.0, 0.4])
    by_gn = GaussNewton(CholeskySolver(), tol=1e-6).minimize(problem, start)
    by_lm = LevenbergMarquardt(CholeskySolver(), tol=1e-6).minimize(problem, start)

    assert by_gn.converged and by_lm.converged
    assert by_lm.iterations == by_gn.iterations
    np.testing.assert_allclose(by_lm.x, by_gn.x, rtol=1e-6, atol=1e-8)


def test_levenberg_marquardt_uses_the_injected_solver() -> None:
    solver = CountingSolver(CholeskySolver())
    LevenbergMarquardt(solver, max_iter=300).minimize(_exp_problem(), np.array([1.0, 10.0, 1.0]))
    assert solver.calls > 0


def test_levenberg_marquardt_matches_scipy() -> None:
    """The oracle. `scipy` lives in the dev extras and may be used from tests only."""
    optimize = pytest.importorskip("scipy.optimize")
    problem = _exp_problem(noise=0.02)
    for start in (np.array([1.0, 1.0, 1.0]), np.array([1.0, 10.0, 1.0])):
        marquardt = LevenbergMarquardt(CholeskySolver(), tol=1e-10, max_iter=500)
        mine = marquardt.minimize(problem, start)
        reference = optimize.least_squares(
            problem.residuals, start, jac=problem.jacobian, method="lm",
            xtol=1e-15, ftol=1e-15, gtol=1e-15,
        )
        np.testing.assert_allclose(mine.x, reference.x, rtol=1e-8, atol=1e-10)


def test_the_same_two_classes_fit_a_different_model() -> None:
    """`GaussianPeak` is a change of data, not of code — the payoff of the interface."""
    t = np.linspace(-3.0, 5.0, 80)
    y = 3.0 * np.exp(-((t - 1.2) ** 2) / (2 * 0.8**2))
    result = LevenbergMarquardt(CholeskySolver(), max_iter=300).minimize(
        GaussianPeak(t, y), np.array([1.0, 0.0, 1.0])
    )
    assert result.converged is True
    np.testing.assert_allclose(np.abs(result.x), [3.0, 1.2, 0.8], atol=1e-6)


def test_levenberg_marquardt_converges_to_a_worse_minimum() -> None:
    """`converged=True` means the gradient is small *here*. It never means "best".

    Starting the frequency at ω = 12 instead of the true 3, LM reports success after 37
    iterations at a cost 178 times worse than the answer found from ω = 3. Both points
    are genuine local minima — ‖Jᵀr‖ is below 1e-8 at each — so no stopping rule could
    tell them apart. Only the cost column can, and the only defence is a grid of starts.
    """
    problem = _sinusoid()
    marquardt = LevenbergMarquardt(CholeskySolver(), max_iter=300)
    good = marquardt.minimize(problem, np.array([1.5, 3.0, 0.0]))
    bad = marquardt.minimize(problem, np.array([1.5, 12.0, 0.0]))

    assert good.converged is True and bad.converged is True
    assert bad.value > 100.0 * good.value
    assert bad.grad_norm <= 1e-8


def test_the_sign_of_omega_is_not_identifiable() -> None:
    """A separate lesson that looks like the same one and is not.

    (a, ω, φ) and (−a, −ω, −φ) describe the *same* curve: −a·sin(−ωt − φ) = a·sin(ωt + φ)
    for every t. So two different parameter vectors give identical predictions and
    identical cost, the minimum is not unique, and no optimizer can prefer one of them.
    The fix belongs in the model — constrain ω > 0 — not in the optimizer.
    """
    problem = _sinusoid()
    marquardt = LevenbergMarquardt(CholeskySolver(), max_iter=300)
    good = marquardt.minimize(problem, np.array([1.5, 3.0, 0.0]))
    flipped = -good.x
    assert not np.allclose(flipped, good.x)
    assert _cost(problem, flipped) == pytest.approx(good.value, rel=1e-12)
