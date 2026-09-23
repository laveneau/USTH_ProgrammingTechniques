"""Part 1 — forward mode with dual numbers."""
import dataclasses
import math
import unittest

from autodiff import Dual, cos, derivative, exp, log, sin


class TestDualArithmetic(unittest.TestCase):
    def test_default_derivative_is_zero(self) -> None:
        self.assertEqual(Dual(3.0), Dual(3.0, 0.0))

    def test_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Dual(1.0, 1.0).val = 2.0            # type: ignore[misc]

    def test_add(self) -> None:
        self.assertEqual(Dual(1, 2) + Dual(3, 4), Dual(4, 6))

    def test_add_with_numbers_on_both_sides(self) -> None:
        self.assertEqual(Dual(1, 2) + 5, Dual(6, 2))
        self.assertEqual(5 + Dual(1, 2), Dual(6, 2))       # needs __radd__

    def test_neg_and_sub(self) -> None:
        self.assertEqual(-Dual(1, 2), Dual(-1, -2))
        self.assertEqual(Dual(5, 1) - Dual(2, 3), Dual(3, -2))
        self.assertEqual(Dual(5, 1) - 2, Dual(3, 1))
        self.assertEqual(10 - Dual(5, 1), Dual(5, -1))     # needs __rsub__

    def test_mul_is_the_product_rule(self) -> None:
        # (a + bε)(c + dε) = ac + (ad + bc)ε
        self.assertEqual(Dual(2, 3) * Dual(4, 5), Dual(8, 22))
        self.assertEqual(3 * Dual(2, 1), Dual(6, 3))       # needs __rmul__

    def test_div_is_the_quotient_rule(self) -> None:
        q = Dual(6, 1) / Dual(2, 1)                          # (u/v)' = (u'v - uv')/v²
        self.assertAlmostEqual(q.val, 3.0)
        self.assertAlmostEqual(q.der, (1 * 2 - 6 * 1) / 4)
        r = 1 / Dual(2, 1)                                   # needs __rtruediv__
        self.assertAlmostEqual(r.val, 0.5)
        self.assertAlmostEqual(r.der, -0.25)

    def test_pow(self) -> None:
        p = Dual(3, 1) ** 2
        self.assertAlmostEqual(p.val, 9.0)
        self.assertAlmostEqual(p.der, 6.0)

    def test_unsupported_operand_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Dual(1, 1) + "two"                              # type: ignore[operator]


class TestDualFunctions(unittest.TestCase):
    def test_exp_log_sin_cos(self) -> None:
        x = 0.7
        self.assertAlmostEqual(derivative(exp, x), math.exp(x))
        self.assertAlmostEqual(derivative(log, x), 1 / x)
        self.assertAlmostEqual(derivative(sin, x), math.cos(x))
        self.assertAlmostEqual(derivative(cos, x), -math.sin(x))

    def test_chain_rule(self) -> None:
        # d/dx sin(x²) = 2x cos(x²)
        x = 1.3
        self.assertAlmostEqual(derivative(lambda t: sin(t * t), x), 2 * x * math.cos(x * x))

    def test_lecture_example(self) -> None:
        # f(x) = x·eˣ  ->  f'(1) = 2e
        self.assertAlmostEqual(derivative(lambda t: t * exp(t), 1.0), 2 * math.e)

    def test_constant_function(self) -> None:
        self.assertEqual(derivative(lambda t: 42.0, 1.0), 0.0)

    def test_derivative_through_a_loop(self) -> None:
        # Babylonian square root of a: we differentiate a *program*, not a formula
        def babylonian_sqrt(a: Dual) -> Dual:
            x = a
            for _ in range(20):
                x = (x + a / x) / 2
            return x

        self.assertAlmostEqual(derivative(babylonian_sqrt, 2.0), 1 / (2 * math.sqrt(2)))


if __name__ == "__main__":
    unittest.main()
