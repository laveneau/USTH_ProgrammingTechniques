"""Elementary functions working on floats, Dual and Var alike (PROVIDED — do not modify).

This is duck typing (Lecture 2): each function delegates to the method of the same
name when its argument is a Dual or a Var, and to the `math` module otherwise.
The constrained TypeVar tells mypy that the output has the same type as the input.
"""
from __future__ import annotations

import math
from typing import TypeVar

from autodiff.dual import Dual
from autodiff.var import Var

Num = TypeVar("Num", float, Dual, Var)


def exp(x: Num) -> Num:
    if isinstance(x, (Dual, Var)):
        return x.exp()
    return math.exp(x)


def log(x: Num) -> Num:
    if isinstance(x, (Dual, Var)):
        return x.log()
    return math.log(x)


def sin(x: Num) -> Num:
    if isinstance(x, (Dual, Var)):
        return x.sin()
    return math.sin(x)


def cos(x: Num) -> Num:
    if isinstance(x, (Dual, Var)):
        return x.cos()
    return math.cos(x)
