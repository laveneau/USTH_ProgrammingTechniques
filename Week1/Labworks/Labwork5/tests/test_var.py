"""Part 2 — reverse mode (backpropagation) on scalars."""
import math
import unittest

from autodiff import Var, cos, derivative, exp, log, sin


class TestVarForwardValues(unittest.TestCase):
    def test_values_are_computed(self) -> None:
        x, y = Var(2.0), Var(3.0)
        self.assertAlmostEqual((x * y + x).value, 8.0)
        self.assertAlmostEqual((x / y - 1).value, 2 / 3 - 1)
        self.assertAlmostEqual((2 - x).value, 0.0)
        self.assertAlmostEqual((1 / x).value, 0.5)

    def test_unsupported_operand_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Var(1.0) * "three"                              # type: ignore[operator]


class TestBackward(unittest.TestCase):
    def test_lecture_example(self) -> None:
        # L = x·y + x at (2, 3): x̄ = y + 1 = 4, ȳ = x = 2
        x, y = Var(2.0), Var(3.0)
        (x * y + x).backward()
        self.assertAlmostEqual(x.grad, 4.0)
        self.assertAlmostEqual(y.grad, 2.0)

    def test_shared_node_accumulates(self) -> None:
        # x used twice: d(x·x)/dx = 2x
        x = Var(3.0)
        (x * x).backward()
        self.assertAlmostEqual(x.grad, 6.0)

    def test_diamond_graph(self) -> None:
        # a feeds b and c, both feed d: gradients must be summed once each
        a = Var(2.0)
        b = a * 3
        c = a ** 2
        (b + c).backward()                                   # d = 3a + a²  ->  3 + 2a = 7
        self.assertAlmostEqual(a.grad, 7.0)

    def test_every_operation(self) -> None:
        x = Var(1.5)
        (-x + 2 * x - 1 / x + x ** 3 - (4 - x) / 2).backward()
        expected = -1 + 2 + 1 / 1.5 ** 2 + 3 * 1.5 ** 2 + 0.5
        self.assertAlmostEqual(x.grad, expected)

    def test_elementary_functions(self) -> None:
        for f, df in ((exp, math.exp), (log, lambda t: 1 / t),
                      (sin, math.cos), (cos, lambda t: -math.sin(t))):
            x = Var(0.8)
            f(x).backward()
            self.assertAlmostEqual(x.grad, df(0.8))

    def test_long_graph_does_not_overflow_the_stack(self) -> None:
        # a recursive topological sort fails here (RecursionError)
        x = Var(1.0)
        y = x
        for _ in range(10_000):
            y = y + x
        y.backward()
        self.assertAlmostEqual(x.grad, 10_001.0)

    def test_reverse_agrees_with_forward(self) -> None:
        def f(t):                                            # works for Dual and Var
            return exp(sin(t)) * t / (1 + t * t)

        x = Var(0.9)
        f(x).backward()
        self.assertAlmostEqual(x.grad, derivative(f, 0.9))


if __name__ == "__main__":
    unittest.main()
