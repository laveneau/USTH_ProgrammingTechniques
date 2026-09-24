"""[DAY 2] The one descent loop, and the pieces it is composed from.

Read these as the specification of Exercise 1 and Exercise 2. Everything here is either
a value you can work out on paper or a property the docstrings promise.
"""

import numpy as np
import pytest

from optlab.errors import LineSearchFailed
from optlab.interfaces import IObjective
from optlab.linesearch import Armijo, FixedStep
from optlab.observers import History
from optlab.optimizers import DescentOptimizer, HeavyBall, SteepestDescent
from optlab.problems import Quadratic
from optlab.results import StepEvent
from optlab.stopping import AnyOf, GradientNormBelow, MaxIterations
from optlab.types import Vec

pytestmark = pytest.mark.day2


class Squares(IObjective):
    """f(x) = ‖x‖², so ∇f = 2x. The hand example from the lecture."""

    def value(self, x: Vec) -> float:
        return float(x @ x)

    def gradient(self, x: Vec) -> Vec:
        return 2.0 * x


class Inconsistent(IObjective):
    """A flat value with a non-zero gradient: no step can ever decrease it.

    Not a realistic objective — it exists to drive `Armijo` to exhaust its budget, which
    is the only way to check that `LineSearchFailed` is raised and then caught.
    """

    def value(self, x: Vec) -> float:
        return 0.0

    def gradient(self, x: Vec) -> Vec:
        return np.ones_like(x)


def _event(iteration: int = 1, grad_norm: float = 1.0) -> StepEvent:
    return StepEvent(iteration=iteration, x=np.zeros(1), value=0.0,
                     grad_norm=grad_norm, step_size=1.0)


# --------------------------------------------------------------------------- #
# Exercise 1 — stopping criteria
# --------------------------------------------------------------------------- #


def test_gradient_norm_below_fires_at_the_threshold() -> None:
    stop = GradientNormBelow(tol=1e-6)
    assert stop.should_stop(_event(grad_norm=1e-7))
    assert stop.should_stop(_event(grad_norm=1e-6))      # <=, not <
    assert not stop.should_stop(_event(grad_norm=1e-5))


def test_max_iterations_fires_on_the_budget() -> None:
    stop = MaxIterations(max_iter=10)
    assert not stop.should_stop(_event(iteration=9))
    assert stop.should_stop(_event(iteration=10))


def test_any_of_fires_when_either_does() -> None:
    stop = AnyOf(GradientNormBelow(1e-6), MaxIterations(10))
    assert stop.should_stop(_event(iteration=1, grad_norm=1e-9))   # convergence only
    assert stop.should_stop(_event(iteration=10, grad_norm=1.0))   # budget only
    assert not stop.should_stop(_event(iteration=1, grad_norm=1.0))


# --------------------------------------------------------------------------- #
# Exercise 1 — observer
# --------------------------------------------------------------------------- #


def test_history_records_what_it_is_given() -> None:
    hist = History()
    for i in (1, 2, 3):
        hist.on_step(StepEvent(iteration=i, x=np.zeros(2), value=float(i),
                               grad_norm=1.0 / i, step_size=0.5))
    assert len(hist.events) == 3
    assert hist.values == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(hist.grad_norms, [1.0, 0.5, 1 / 3])


# --------------------------------------------------------------------------- #
# Exercise 1 — direction rules
# --------------------------------------------------------------------------- #


def test_steepest_descent_is_minus_the_gradient() -> None:
    g = np.array([2.0, -3.0])
    np.testing.assert_allclose(SteepestDescent().direction(Squares(), np.zeros(2), g), -g)


def test_heavy_ball_has_no_momentum_on_the_first_call() -> None:
    g = np.array([1.0, -1.0])
    np.testing.assert_allclose(HeavyBall(0.9).direction(Squares(), np.zeros(2), g), -g)


def test_heavy_ball_remembers_the_previous_direction() -> None:
    """The state is the point: an `IDirectionRule` is an object, not a function."""
    rule = HeavyBall(beta=0.5)
    g = np.array([1.0, -1.0])
    first = rule.direction(Squares(), np.zeros(2), g)
    second = rule.direction(Squares(), np.zeros(2), g)
    np.testing.assert_allclose(second, -g + 0.5 * first)


# --------------------------------------------------------------------------- #
# Exercise 1 / 2 — line searches
# --------------------------------------------------------------------------- #


def test_fixed_step_returns_its_alpha_whatever_it_is_handed() -> None:
    x = np.array([3.0])
    assert FixedStep(0.07).step(Squares(), x, np.array([6.0]), np.array([-6.0])) == 0.07


def test_armijo_hand_example_from_the_lecture() -> None:
    """f(x) = x² at x = 1: α = 1 lands back at 1 and is rejected; α = 0.5 is accepted."""
    x, g = np.array([1.0]), np.array([2.0])
    assert Armijo(alpha0=1.0, rho=0.5).step(Squares(), x, g, -g) == pytest.approx(0.5)


def test_armijo_returns_a_step_satisfying_sufficient_decrease() -> None:
    x, g = np.array([1.0, -2.0]), None
    obj = Squares()
    g = obj.gradient(x)
    d = -g
    alpha = Armijo().step(obj, x, g, d)
    assert obj.value(x + alpha * d) <= obj.value(x) + 1e-4 * alpha * float(g @ d)


def test_armijo_raises_when_no_step_can_work() -> None:
    obj = Inconsistent()
    x = np.array([1.0])
    g = obj.gradient(x)
    with pytest.raises(LineSearchFailed):
        Armijo(max_backtracks=5).step(obj, x, g, -g)


def test_armijo_rejects_a_direction_that_does_not_descend() -> None:
    """A non-descent direction is the caller's bug, not something to backtrack around."""
    obj = Squares()
    x = np.array([1.0])
    g = obj.gradient(x)
    with pytest.raises(ValueError):
        Armijo().step(obj, x, g, +g)          # uphill


# --------------------------------------------------------------------------- #
# Exercise 1 / 2 — the loop itself
# --------------------------------------------------------------------------- #


def _descend(direction, line_search, stop, x0, problem=None):
    hist = History()
    opt = DescentOptimizer(direction, line_search, stop, observers=[hist])
    return opt.minimize(problem or Quadratic.ill_conditioned(2, 10.0), x0), hist


def test_loop_emits_exactly_one_event_per_iteration() -> None:
    result, hist = _descend(SteepestDescent(), Armijo(), MaxIterations(7), np.ones(2))
    assert len(hist.events) == 7
    assert [e.iteration for e in hist.events] == list(range(1, 8))
    assert result.iterations == 7


def test_event_describes_where_the_step_landed() -> None:
    """`x`, `value` and `grad_norm` are built after moving, not before."""
    problem = Quadratic.ill_conditioned(2, 10.0)
    _, hist = _descend(SteepestDescent(), Armijo(), MaxIterations(3), np.ones(2), problem)
    for event in hist.events:
        assert event.value == pytest.approx(problem.value(event.x))
        assert event.grad_norm == pytest.approx(np.linalg.norm(problem.gradient(event.x)))


def test_converged_is_true_only_on_a_real_convergence_test() -> None:
    result, _ = _descend(SteepestDescent(), Armijo(),
                         AnyOf(GradientNormBelow(1e-8), MaxIterations(5000)), np.ones(2))
    assert result.converged
    assert result.grad_norm <= 1e-8


def test_running_out_of_budget_is_not_convergence() -> None:
    result, _ = _descend(SteepestDescent(), Armijo(),
                         AnyOf(GradientNormBelow(1e-12), MaxIterations(3)), np.ones(2))
    assert not result.converged
    assert result.iterations == 3
    assert result.message


def test_a_failed_line_search_is_reported_not_raised() -> None:
    """The optimizer catches `LineSearchFailed` and reports it; it must not crash."""
    opt = DescentOptimizer(SteepestDescent(), Armijo(max_backtracks=5), MaxIterations(10))
    result = opt.minimize(Inconsistent(), np.array([1.0]))
    assert not result.converged
    assert result.message


def test_the_loop_actually_minimizes() -> None:
    result, _ = _descend(SteepestDescent(), Armijo(),
                         AnyOf(GradientNormBelow(1e-10), MaxIterations(20000)), np.ones(2))
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-8)


def test_swapping_the_line_search_needs_no_change_to_the_loop() -> None:
    """Open/closed, made executable: the same loop, one collaborator replaced."""
    stop = AnyOf(GradientNormBelow(1e-8), MaxIterations(20000))
    fixed, _ = _descend(SteepestDescent(), FixedStep(0.05), stop, np.ones(2))
    armijo, _ = _descend(SteepestDescent(), Armijo(),
                         AnyOf(GradientNormBelow(1e-8), MaxIterations(20000)), np.ones(2))
    assert fixed.converged and armijo.converged
    np.testing.assert_allclose(fixed.x, armijo.x, atol=1e-6)


def test_momentum_beats_steepest_descent_on_an_ill_conditioned_quadratic() -> None:
    """The measurement of Exercise 3, as a test: √κ instead of κ."""
    kappa, mu, L = 100.0, 1.0, 100.0
    problem = Quadratic.ill_conditioned(2, kappa)
    beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    stop = lambda: AnyOf(GradientNormBelow(1e-8), MaxIterations(20000))
    sd, _ = _descend(SteepestDescent(), FixedStep(2.0 / (L + mu)), stop(), np.ones(2), problem)
    hb, _ = _descend(HeavyBall(beta), FixedStep(4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2),
                     stop(), np.ones(2), problem)
    assert sd.converged and hb.converged
    assert hb.iterations < sd.iterations / 4
