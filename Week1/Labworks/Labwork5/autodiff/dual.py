"""Forward mode: dual numbers a + b·ε with ε² = 0.

Part 1 of the project. Replace every `raise NotImplementedError` by an implementation.
`__add__` is given as a worked example.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Dual:
    """A dual number `val + der·ε`: a value and its derivative travelling together."""

    val: float
    der: float = 0.0

    @staticmethod
    def lift(x: Dual | float) -> Dual:
        """Turn a plain number into a constant dual number (its derivative is 0)."""
        if isinstance(x, Dual):
            return x
        return Dual(float(x), 0.0)

    # ---- arithmetic ------------------------------------------------------------
    def __add__(self, other: Dual | float) -> Dual:
        """Worked example: (a + bε) + (c + dε) = (a + c) + (b + d)ε."""
        if not isinstance(other, (Dual, int, float)):
            return NotImplemented          # lets Python raise the proper TypeError
        o = Dual.lift(other)
        return Dual(self.val + o.val, self.der + o.der)

    def __radd__(self, other: float) -> Dual:
        """Called for `2 + x` when `x` is a Dual (addition is commutative)."""
        raise NotImplementedError("TODO: Dual.__radd__")

    def __neg__(self) -> Dual:
        raise NotImplementedError("TODO: Dual.__neg__")

    def __sub__(self, other: Dual | float) -> Dual:
        raise NotImplementedError("TODO: Dual.__sub__ (hint: reuse + and unary -)")

    def __rsub__(self, other: float) -> Dual:
        """Called for `2 - x`: careful, subtraction is NOT commutative."""
        raise NotImplementedError("TODO: Dual.__rsub__")

    def __mul__(self, other: Dual | float) -> Dual:
        """(a + bε)(c + dε) = ac + (ad + bc)ε  — the product rule!"""
        raise NotImplementedError("TODO: Dual.__mul__")

    def __rmul__(self, other: float) -> Dual:
        raise NotImplementedError("TODO: Dual.__rmul__")

    def __truediv__(self, other: Dual | float) -> Dual:
        """(u/v)' = (u'v − uv')/v²  — the quotient rule."""
        raise NotImplementedError("TODO: Dual.__truediv__")

    def __rtruediv__(self, other: float) -> Dual:
        raise NotImplementedError("TODO: Dual.__rtruediv__")

    def __pow__(self, exponent: float) -> Dual:
        """Only a constant (float) exponent: (uᵏ)' = k·uᵏ⁻¹·u'."""
        raise NotImplementedError("TODO: Dual.__pow__")

    # ---- elementary functions (chain rule: (f∘g)' = f'(g)·g') ------------------
    def exp(self) -> Dual:
        raise NotImplementedError("TODO: Dual.exp")

    def log(self) -> Dual:
        raise NotImplementedError("TODO: Dual.log")

    def sin(self) -> Dual:
        raise NotImplementedError("TODO: Dual.sin")

    def cos(self) -> Dual:
        raise NotImplementedError("TODO: Dual.cos")


def derivative(f: Callable[[Dual], Dual | float], x: float) -> float:
    """Return f'(x) with a single forward sweep.

    Hint: evaluate f at the dual number x + 1·ε. Beware, f may return a plain
    float (a constant function): its derivative is then 0.
    """
    raise NotImplementedError("TODO: derivative")
