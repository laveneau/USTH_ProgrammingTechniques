"""Gradients of functions of several variables."""
from __future__ import annotations

from typing import Callable, Sequence

from autodiff.dual import Dual
from autodiff.var import Var


def numerical_gradient(f: Callable[[list[float]], float], x: Sequence[float],
                       h: float = 1e-6) -> list[float]:
    """Central finite differences (PROVIDED): an approximation, used as a reference."""
    grad: list[float] = []
    for i in range(len(x)):
        plus = [float(v) for v in x]
        minus = [float(v) for v in x]
        plus[i] += h
        minus[i] -= h
        grad.append((f(plus) - f(minus)) / (2 * h))
    return grad


def gradient_forward(f: Callable[[list[Dual]], Dual], x: Sequence[float]) -> list[float]:
    """Exact gradient with forward mode.

    Hint: one sweep per input variable i, seeding a derivative of 1 on x[i] and 0
    elsewhere. How many sweeps for a function of 1 000 variables?
    """
    raise NotImplementedError("TODO: gradient_forward")


def gradient_reverse(f: Callable[[list[Var]], Var], x: Sequence[float]) -> list[float]:
    """Exact gradient with reverse mode: a single backward sweep.

    Hint: wrap the inputs into Var leaves, evaluate f, call backward, read the grads.
    """
    raise NotImplementedError("TODO: gradient_reverse")
