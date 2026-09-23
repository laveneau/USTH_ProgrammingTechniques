"""Part 3 — gradients of functions of several variables."""
import unittest

from autodiff import exp, gradient_forward, gradient_reverse, numerical_gradient


def rosenbrock(x):
    """Rosenbrock in dimension n; minimum 0 at (1, ..., 1)."""
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def rosenbrock_gradient(x):
    """Hand-written gradient of `rosenbrock`."""
    n = len(x)
    g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return g


def least_squares(data):
    """½ Σ (w·xᵢ − yᵢ)² as a function of w (a preview of Week 2)."""
    def loss(w):
        return 0.5 * sum((sum(wj * xj for wj, xj in zip(w, x)) - y) ** 2 for x, y in data)
    return loss


class TestGradients(unittest.TestCase):
    point = [-1.2, 1.0, 0.5, 2.0]

    def assert_lists_close(self, a, b, places=6) -> None:
        self.assertEqual(len(a), len(b))
        for u, v in zip(a, b):
            self.assertAlmostEqual(u, v, places=places)

    def test_numerical_gradient_is_close(self) -> None:
        self.assert_lists_close(numerical_gradient(rosenbrock, self.point),
                                rosenbrock_gradient(self.point), places=3)

    def test_forward_mode_is_exact(self) -> None:
        self.assert_lists_close(gradient_forward(rosenbrock, self.point),
                                rosenbrock_gradient(self.point), places=9)

    def test_reverse_mode_is_exact(self) -> None:
        self.assert_lists_close(gradient_reverse(rosenbrock, self.point),
                                rosenbrock_gradient(self.point), places=9)

    def test_least_squares(self) -> None:
        data = [([1.0, 2.0], 5.0), ([3.0, -1.0], 1.0), ([0.5, 0.5], 2.0)]
        w = [0.3, -0.7]
        # hand gradient: Σ (w·xᵢ − yᵢ) xᵢ
        expected = [0.0, 0.0]
        for x, y in data:
            r = w[0] * x[0] + w[1] * x[1] - y
            expected = [expected[0] + r * x[0], expected[1] + r * x[1]]
        self.assert_lists_close(gradient_forward(least_squares(data), w), expected)
        self.assert_lists_close(gradient_reverse(least_squares(data), w), expected)

    def test_forward_and_reverse_agree(self) -> None:
        def f(x):
            return exp(x[0] * x[1]) + x[2] / (1 + x[0] ** 2)
        p = [0.3, -0.4, 2.0]
        self.assert_lists_close(gradient_forward(f, p), gradient_reverse(f, p), places=12)


if __name__ == "__main__":
    unittest.main()
