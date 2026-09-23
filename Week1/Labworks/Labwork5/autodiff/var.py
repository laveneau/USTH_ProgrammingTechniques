"""Reverse mode (backpropagation) on scalars.

Part 2 of the project. Replace every `raise NotImplementedError` by an implementation.
`__add__` is given as a worked example.
"""
from __future__ import annotations

import math


class Var:
    """A node of the computation graph.

    Each node stores its value, its gradient (the adjoint ∂output/∂node, filled by
    `backward`) and its parents together with the *local* derivative
    ∂node/∂parent.
    """

    def __init__(self, value: float, parents: tuple[tuple[Var, float], ...] = ()) -> None:
        self.value = float(value)
        self.grad = 0.0
        self.parents = parents

    def __repr__(self) -> str:
        return f"Var(value={self.value}, grad={self.grad})"

    @staticmethod
    def lift(x: Var | float) -> Var:
        """Turn a plain number into a constant leaf of the graph."""
        if isinstance(x, Var):
            return x
        return Var(float(x))

    # ---- arithmetic: each operation records (parent, local derivative) ---------
    def __add__(self, other: Var | float) -> Var:
        """Worked example: z = x + y, so ∂z/∂x = 1 and ∂z/∂y = 1."""
        if not isinstance(other, (Var, int, float)):
            return NotImplemented
        o = Var.lift(other)
        return Var(self.value + o.value, ((self, 1.0), (o, 1.0)))

    def __radd__(self, other: float) -> Var:
        raise NotImplementedError("TODO: Var.__radd__")

    def __neg__(self) -> Var:
        raise NotImplementedError("TODO: Var.__neg__")

    def __sub__(self, other: Var | float) -> Var:
        raise NotImplementedError("TODO: Var.__sub__")

    def __rsub__(self, other: float) -> Var:
        raise NotImplementedError("TODO: Var.__rsub__")

    def __mul__(self, other: Var | float) -> Var:
        """z = x·y: ∂z/∂x = y and ∂z/∂y = x."""
        raise NotImplementedError("TODO: Var.__mul__")

    def __rmul__(self, other: float) -> Var:
        raise NotImplementedError("TODO: Var.__rmul__")

    def __truediv__(self, other: Var | float) -> Var:
        raise NotImplementedError("TODO: Var.__truediv__")

    def __rtruediv__(self, other: float) -> Var:
        raise NotImplementedError("TODO: Var.__rtruediv__")

    def __pow__(self, exponent: float) -> Var:
        raise NotImplementedError("TODO: Var.__pow__ (constant exponent only)")

    # ---- elementary functions ----------------------------------------------------
    def exp(self) -> Var:
        raise NotImplementedError("TODO: Var.exp")

    def log(self) -> Var:
        raise NotImplementedError("TODO: Var.log")

    def sin(self) -> Var:
        raise NotImplementedError("TODO: Var.sin")

    def cos(self) -> Var:
        raise NotImplementedError("TODO: Var.cos")

    # ---- backpropagation ---------------------------------------------------------
    def topological_order(self) -> list[Var]:
        """All nodes of the graph, each node appearing AFTER all of its parents.

        Hint: a depth-first search. Write it WITHOUT recursion (use an explicit
        stack): a recursive version overflows Python's stack on long graphs.
        Mind nodes reachable by several paths: each must appear only once.
        """
        raise NotImplementedError("TODO: Var.topological_order")

    def backward(self) -> None:
        """Fill `.grad` of every node with ∂self/∂node.

        Hint: seed self.grad = 1, then visit the nodes from the output back to the
        inputs, and push each node's gradient to its parents with the chain rule.
        Remember that a node may be used several times...
        """
        raise NotImplementedError("TODO: Var.backward")
