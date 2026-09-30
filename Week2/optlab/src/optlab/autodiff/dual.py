"""PROVIDED - forward-mode automatic differentiation over numpy arrays.

Week 1's `Dual` carried a number and its derivative. This one carries an array and its
derivative, and that is the only change: broadcasting needs no correction in the forward
direction, so every operator rule is the Week 1 rule verbatim.

Forward mode costs one sweep per input coordinate, so it loses to reverse mode for a loss
of many parameters. It wins in the opposite shape - few inputs, many outputs - which is
exactly a Jacobian of a curve-fitting model with three parameters (day 5).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, Union

import numpy as np

from ..types import Mat, Vec

Operand: TypeAlias = Union["Dual", Vec, float, int]


@dataclass(frozen=True)
class Dual:
    """val + der*eps with eps^2 = 0, so that f(a + b*eps) = f(a) + f'(a)*b*eps."""

    val: Vec
    der: Vec

    __array_ufunc__ = None

    @staticmethod
    def lift(x: Operand) -> "Dual":
        if isinstance(x, Dual):
            return x
        v = np.asarray(x, dtype=np.float64)
        return Dual(v, np.zeros_like(v))

    def __add__(self, other: Operand) -> "Dual":
        o = Dual.lift(other)
        return Dual(self.val + o.val, self.der + o.der)

    def __radd__(self, other: Operand) -> "Dual":
        return self + other

    def __neg__(self) -> "Dual":
        return Dual(-self.val, -self.der)

    def __sub__(self, other: Operand) -> "Dual":
        o = Dual.lift(other)
        return Dual(self.val - o.val, self.der - o.der)

    def __rsub__(self, other: Operand) -> "Dual":
        return Dual.lift(other) - self

    def __mul__(self, other: Operand) -> "Dual":
        """(u + u'eps)(v + v'eps) = uv + (u'v + uv')eps - the product rule falls out."""
        o = Dual.lift(other)
        return Dual(self.val * o.val, self.der * o.val + self.val * o.der)

    def __rmul__(self, other: Operand) -> "Dual":
        return self * other

    def __truediv__(self, other: Operand) -> "Dual":
        o = Dual.lift(other)
        return Dual(self.val / o.val,
                    (self.der * o.val - self.val * o.der) / (o.val * o.val))

    def __rtruediv__(self, other: Operand) -> "Dual":
        return Dual.lift(other) / self

    def __pow__(self, exponent: float) -> "Dual":
        return Dual(self.val**exponent,
                    exponent * self.val ** (exponent - 1) * self.der)

    def __matmul__(self, other: Operand) -> "Dual":
        o = Dual.lift(other)
        return Dual(self.val @ o.val, self.der @ o.val + self.val @ o.der)

    def __rmatmul__(self, other: Operand) -> "Dual":
        return Dual.lift(other) @ self

    def __getitem__(self, key: int) -> "Dual":
        """Index both halves at once, so a model can be written `w[0]*exp(-w[1]*t)`."""
        return Dual(self.val[key], self.der[key])

    def exp(self) -> "Dual":
        value = np.exp(self.val)
        return Dual(value, value * self.der)

    def log(self) -> "Dual":
        return Dual(np.log(self.val), self.der / self.val)

    def sin(self) -> "Dual":
        return Dual(np.sin(self.val), np.cos(self.val) * self.der)

    def cos(self) -> "Dual":
        return Dual(np.cos(self.val), -np.sin(self.val) * self.der)

    def sum(self) -> "Dual":
        return Dual(np.asarray(self.val.sum()), np.asarray(self.der.sum()))



def _seed(x: Vec, i: int) -> Dual:
    """`x` carrying the i-th basis vector as its derivative: one forward sweep."""
    der = np.zeros_like(x)
    der[i] = 1.0
    return Dual(x, der)


def derivative(f: Callable[[Dual], Dual], x: float) -> float:
    """f'(x) for a scalar function, by one forward sweep."""
    out = f(Dual(np.asarray(x, dtype=np.float64), np.asarray(1.0)))
    return float(out.der)


def gradient_forward(f: Callable[[Dual], Dual], x: Vec) -> Vec:
    """The full gradient, at a cost of n sweeps - which is why reverse mode exists."""
    return np.array([float(f(_seed(x, i)).der) for i in range(x.size)], dtype=np.float64)


def jacobian_forward(f: Callable[[Dual], Dual], x: Vec) -> Mat:
    """[DAY 5] The Jacobian of a vector-valued `f`, one column per input coordinate.

    Forward mode is the right tool here: a curve-fitting model has a handful of
    parameters and many residuals, so n sweeps is cheap and each sweep fills a column.
    """
    return np.column_stack([f(_seed(x, i)).der for i in range(x.size)])
