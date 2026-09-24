"""Every `IDirectionRule` must return a descent direction: gᵀd < 0 whenever g ≠ 0.

The same test over every rule, which is what makes swapping one for another safe. Add
your rules to `RULES` as you write them; the day-4 entries are skipped until the classes
they need exist, so this file stays green on day 2 without pretending day 4 is done.

Each entry is a *factory*, so a constructor that is still a stub fails its own test
rather than breaking collection of the whole suite.
"""

from collections.abc import Callable

import numpy as np
import pytest

from optlab.interfaces import IDirectionRule
from optlab.optimizers import HeavyBall, SteepestDescent
from optlab.problems import Quadratic, Rosenbrock

DAY2_RULES: dict[str, Callable[[], IDirectionRule]] = {
    "steepest": SteepestDescent,
    "heavy_ball": lambda: HeavyBall(0.9),
}

PROBLEMS = {
    "quadratic": (lambda: Quadratic.ill_conditioned(4, 50.0), np.array([1.0, -2.0, 0.5, 3.0])),
    "rosenbrock": (Rosenbrock, np.array([-1.2, 1.0])),
}

CASES = [(r, p) for r in DAY2_RULES for p in PROBLEMS]


@pytest.mark.contract
@pytest.mark.day2
@pytest.mark.parametrize("rule_name,problem_name", CASES)
def test_direction_is_a_descent_direction(rule_name: str, problem_name: str) -> None:
    """gᵀd < 0, or the line search is being asked to walk uphill."""
    make_problem, x = PROBLEMS[problem_name]
    problem = make_problem()
    rule = DAY2_RULES[rule_name]()
    g = problem.gradient(x)
    d = rule.direction(problem, x, g)
    assert float(g @ d) < 0.0


@pytest.mark.contract
@pytest.mark.day2
@pytest.mark.parametrize("rule_name", list(DAY2_RULES), ids=list(DAY2_RULES))
def test_direction_has_the_shape_of_the_gradient(rule_name: str) -> None:
    problem = Quadratic.ill_conditioned(4, 50.0)
    x = np.array([1.0, -2.0, 0.5, 3.0])
    g = problem.gradient(x)
    assert DAY2_RULES[rule_name]().direction(problem, x, g).shape == g.shape


@pytest.mark.contract
@pytest.mark.day2
def test_a_stateless_rule_is_reproducible() -> None:
    """`SteepestDescent` keeps no state, so asking twice must give the same answer."""
    problem = Quadratic.ill_conditioned(3, 10.0)
    x = np.array([1.0, 1.0, 1.0])
    g = problem.gradient(x)
    rule = SteepestDescent()
    np.testing.assert_allclose(rule.direction(problem, x, g), rule.direction(problem, x, g))
