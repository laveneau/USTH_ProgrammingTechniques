"""The three two-dimensional algebras — the Week 1 bonus, kept outside the package.

Numbers a + b·t with t² = s:

    s = -1   complex      i² = -1    rotation
    s =  0   dual         e² =  0    shear      <- the one that differentiates exactly
    s = +1   hyperbolic   j² = +1    boost

Only s = 0 gives an *exact* first derivative. s = -1 gives the complex-step trick, which
is accurate to O(h²) but still an approximation.

This lives in `labs/` rather than `src/optlab/` on purpose: it is a beautiful aside, not
something the optimizers depend on. Keeping it out of the package is the KISS decision
of the week — a generic `Num2` inside `src/` would be an abstraction with exactly one
user.

Optional. Nothing in the week's grading depends on it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Num2:
    """a + b·t where t² = s."""

    a: float
    b: float
    s: int

    def __mul__(self, other: "Num2") -> "Num2":
        """(a + bt)(c + dt) = ac + s·bd + (ad + bc)t."""
        raise NotImplementedError("optional lab")

    def quadratic_form(self) -> float:
        """Q(a + bt) = a² + s·b². Signature (1,1,0) / (1,0,1) / (2,0,0)."""
        raise NotImplementedError("optional lab")
