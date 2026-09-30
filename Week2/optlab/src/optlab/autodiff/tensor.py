"""PROVIDED - reverse-mode automatic differentiation over numpy arrays.

This is the Week 1 project generalized from numbers to arrays: same computation graph,
same backward walk, same chain rule. Three things had to change, and they are the whole
difference between a teaching autodiff and a usable one.

1. A node no longer stores a *local derivative* next to each parent, it stores a
   *function*. For z = x*y the local derivative is just y, and `parent.grad += y * g`
   works. For Z = A @ B there is no array L with `A.grad += L * g`: the rule is
   `A.grad += g @ B.T`. So each edge carries a small closure - its vector-Jacobian
   product - that turns the output adjoint into this parent's contribution.

2. Broadcasting has to be undone on the way back. numpy silently stretches a shape (1,)
   operand across a shape (n,) one going forward, so the adjoint comes back with the
   wrong shape and must be summed over the stretched axes. That is `unbroadcast`, and
   forgetting it is the classic source of a silently wrong gradient.

3. A reduction is needed, because a loss is one number built from arrays. `sum` sends the
   scalar adjoint back out over the input's shape.

Nothing in `src/optlab/` depends on this module. It exists as a second, *exact* gradient
oracle for the tests: finite differences bottom out near 1e-11 (day 1's U-curve), while
this is correct to machine precision.
"""

from collections.abc import Callable
from typing import TypeAlias, Union

import numpy as np

from ..types import Vec

Operand: TypeAlias = Union["Tensor", Vec, float, int]

# An edge of the graph: the parent, and the vector-Jacobian product that pushes an adjoint
# of the child back onto it.
Edge: TypeAlias = tuple["Tensor", Callable[[Vec], Vec]]


def unbroadcast(g: Vec, shape: tuple[int, ...]) -> Vec:
    """Sum `g` back down to `shape`, undoing numpy broadcasting.

    Going forward, numpy stretches a (1,) or missing axis to match its partner. Going
    backward, every stretched copy contributed to the output, so the contributions add:
    a stretched axis is summed away. This is a no-op when no broadcasting happened.
    """
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


class Tensor:
    """A node of the computation graph, holding an array value and its adjoint.

    `data` is the value computed on the way forward; `grad` is d(output)/d(this), filled
    by `backward`. `_edges` records where the value came from, so the graph can be walked
    in reverse.
    """

    __slots__ = ("_edges", "data", "grad")

    # numpy must not try to handle `array + tensor` itself with its own broadcasting
    # rules; setting this to None makes it defer, so Python calls our __radd__ instead.
    __array_ufunc__ = None

    def __init__(self, data: Operand, edges: tuple[Edge, ...] = ()) -> None:
        self.data: Vec = np.asarray(data, dtype=np.float64)
        self.grad: Vec = np.zeros_like(self.data)
        self._edges = edges

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def lift(x: Operand) -> "Tensor":
        """Wrap a constant as a leaf of the graph, and leave a Tensor alone."""
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- elementwise arithmetic ---------------------------------------------------
    # Each of these is the Week 1 rule with the local derivative wrapped in a closure and
    # passed through `unbroadcast`.

    def __add__(self, other: Operand) -> "Tensor":
        """z = x + y, so dz/dx = 1 and dz/dy = 1 - the adjoint passes straight through."""
        o = Tensor.lift(other)
        return Tensor(
            self.data + o.data,
            ((self, lambda g: unbroadcast(g, self.shape)),
             (o, lambda g: unbroadcast(g, o.shape))),
        )

    def __radd__(self, other: Operand) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data, ((self, lambda g: -g),))

    def __sub__(self, other: Operand) -> "Tensor":
        o = Tensor.lift(other)
        return Tensor(
            self.data - o.data,
            ((self, lambda g: unbroadcast(g, self.shape)),
             (o, lambda g: unbroadcast(-g, o.shape))),
        )

    def __rsub__(self, other: Operand) -> "Tensor":
        return Tensor.lift(other) - self

    def __mul__(self, other: Operand) -> "Tensor":
        """z = x*y, so dz/dx = y and dz/dy = x."""
        o = Tensor.lift(other)
        return Tensor(
            self.data * o.data,
            ((self, lambda g: unbroadcast(g * o.data, self.shape)),
             (o, lambda g: unbroadcast(g * self.data, o.shape))),
        )

    def __rmul__(self, other: Operand) -> "Tensor":
        return self * other

    def __truediv__(self, other: Operand) -> "Tensor":
        o = Tensor.lift(other)
        return Tensor(
            self.data / o.data,
            ((self, lambda g: unbroadcast(g / o.data, self.shape)),
             (o, lambda g: unbroadcast(-g * self.data / (o.data * o.data), o.shape))),
        )

    def __rtruediv__(self, other: Operand) -> "Tensor":
        return Tensor.lift(other) / self

    def __pow__(self, exponent: float) -> "Tensor":
        local = exponent * self.data ** (exponent - 1)
        return Tensor(self.data**exponent, ((self, lambda g: g * local),))

    # ---- matrix product -----------------------------------------------------------

    def __matmul__(self, other: Operand) -> "Tensor":
        return _matmul(self, Tensor.lift(other))

    def __rmatmul__(self, other: Operand) -> "Tensor":
        return _matmul(Tensor.lift(other), self)

    def __getitem__(self, key: int) -> "Tensor":
        """Take one coordinate. Its adjoint scatters back into a zero array."""
        shape = self.shape

        def scatter(g: Vec) -> Vec:
            out = np.zeros(shape, dtype=np.float64)
            out[key] = g
            return out

        return Tensor(self.data[key], ((self, scatter),))

    # ---- elementary functions -----------------------------------------------------

    def exp(self) -> "Tensor":
        value = np.exp(self.data)
        return Tensor(value, ((self, lambda g: g * value),))

    def log(self) -> "Tensor":
        return Tensor(np.log(self.data), ((self, lambda g: g / self.data),))

    def sin(self) -> "Tensor":
        return Tensor(np.sin(self.data), ((self, lambda g: g * np.cos(self.data)),))

    def cos(self) -> "Tensor":
        return Tensor(np.cos(self.data), ((self, lambda g: -g * np.sin(self.data)),))

    # ---- reductions ---------------------------------------------------------------

    def sum(self) -> "Tensor":
        """Add every entry. The adjoint is one number, spread back over the input."""
        shape = self.shape
        return Tensor(self.data.sum(), ((self, lambda g: np.broadcast_to(g, shape).copy()),))

    def mean(self) -> "Tensor":
        return self.sum() / float(self.data.size)

    # ---- backpropagation ----------------------------------------------------------

    def _topological_order(self) -> list["Tensor"]:
        """Every node of the graph, each appearing after all of its parents.

        Iterative, not recursive: a recursive walk overflows Python's stack on a deep
        graph. The flag on each stack entry says whether the node's parents were already
        pushed, so a node is appended only on the second visit.
        """
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent, _ in node._edges:
                if id(parent) not in seen:
                    stack.append((parent, False))
        return order

    def backward(self) -> None:
        """Fill `grad` on every node this one was built from.

        Only defined for a scalar output, which is what a loss is; for anything else there
        is no single adjoint to seed the walk with.
        """
        if self.data.ndim != 0:
            raise ValueError(f"backward() needs a scalar output, got shape {self.shape}")
        order = self._topological_order()
        for node in order:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            for parent, vjp in node._edges:
                parent.grad = parent.grad + vjp(node.grad)


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """`a @ b`, with the vector-Jacobian product for each of the four shape cases.

    This is the operation that forced an edge to carry a function rather than an array:
    the adjoint of `a` is built with a *product*, not an elementwise multiplication.
    """
    x, y = a.data, b.data
    value = x @ y

    if x.ndim == 2 and y.ndim == 1:          # (n, p) @ (p,) -> (n,)
        return Tensor(value, ((a, lambda g: np.outer(g, y)), (b, lambda g: x.T @ g)))
    if x.ndim == 2 and y.ndim == 2:          # (n, k) @ (k, m) -> (n, m)
        return Tensor(value, ((a, lambda g: g @ y.T), (b, lambda g: x.T @ g)))
    if x.ndim == 1 and y.ndim == 1:          # (k,) @ (k,) -> scalar
        return Tensor(value, ((a, lambda g: g * y), (b, lambda g: g * x)))
    if x.ndim == 1 and y.ndim == 2:          # (k,) @ (k, m) -> (m,)
        return Tensor(value, ((a, lambda g: y @ g), (b, lambda g: np.outer(x, g))))
    raise ValueError(f"matmul not supported for shapes {x.shape} and {y.shape}")



def autodiff_gradient(f: Callable[[Tensor], Tensor]) -> Callable[[Vec], Vec]:
    """Turn a function written over `Tensor` into its exact gradient function.

    This is the entry point the tests use. `f` must return a scalar `Tensor`; the result
    is a plain numpy gradient of the same shape as the input, exact to machine precision.
    """

    def gradient(x: Vec) -> Vec:
        leaf = Tensor(x)
        out = f(leaf)
        out.backward()
        return leaf.grad

    return gradient

