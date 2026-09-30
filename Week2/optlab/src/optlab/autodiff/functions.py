"""PROVIDED - elementary functions that work in either mode.

`exp(x)` has to mean the same thing whether `x` is a `Dual` (forward mode) or a `Tensor`
(reverse mode), so each one simply calls the method and the constrained type variable
keeps the two apart for the type checker.
"""

from typing import TypeVar

from .dual import Dual
from .tensor import Tensor

Node = TypeVar("Node", Dual, Tensor)


def exp(x: Node) -> Node:
    return x.exp()


def log(x: Node) -> Node:
    return x.log()


def sin(x: Node) -> Node:
    return x.sin()


def cos(x: Node) -> Node:
    return x.cos()
