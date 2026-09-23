"""PLACEHOLDER — replace this file with your Week 1 forward-mode module.

Your Week 1 version already provides everything needed here:

    @dataclass(frozen=True)
    class Dual:
        val: float
        der: float
        # __add__, __radd__, __sub__, __neg__, __mul__, __rmul__, __truediv__, __pow__

    def exp(x): ...
    def log(x): ...
    def derivative(f, x) -> float: ...
    def gradient_forward(f, x) -> Vec: ...       # n sweeps

Copy it over as it stands; it must still pass `mypy --strict` inside this package.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..types import Vec


@dataclass(frozen=True)
class Dual:
    """a + b·ε with ε² = 0, so that f(a + bε) = f(a) + f'(a)·b·ε."""

    val: float
    der: float


def derivative(f: Callable[[Dual], Dual], x: float) -> float:
    """f'(x) by one forward sweep."""
    raise NotImplementedError("copy your Week 1 autodiff/dual.py here")


def gradient_forward(f: Callable[[Vec], Dual], x: Vec) -> Vec:
    """The full gradient, at a cost of n sweeps — which is why reverse mode exists."""
    raise NotImplementedError("copy your Week 1 autodiff/dual.py here")
