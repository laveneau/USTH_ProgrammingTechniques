"""PLACEHOLDER — replace this file with your Week 1 reverse-mode module.

Your Week 1 version already provides everything needed here:

    class Tensor:
        data: Vec
        grad: Vec
        # __add__ __sub__ __mul__ __truediv__ __neg__ __pow__ __matmul__
        # sum(), exp(), log(), backward()

    def autodiff_gradient(f) -> Callable[[Vec], Vec]: ...

`autodiff_gradient` is the one this week uses: from day 1 it is the second oracle in
every gradient test, alongside finite differences.
"""

from collections.abc import Callable

from ..types import Vec


class Tensor:
    """A node in a computation graph, carrying a value and an accumulated adjoint."""

    def __init__(self, data: Vec) -> None:
        raise NotImplementedError("copy your Week 1 autodiff/tensor.py here")


def autodiff_gradient(f: Callable[[Tensor], Tensor]) -> Callable[[Vec], Vec]:
    """Turn a function written over `Tensor` into its exact gradient function."""
    raise NotImplementedError("copy your Week 1 autodiff/tensor.py here")
