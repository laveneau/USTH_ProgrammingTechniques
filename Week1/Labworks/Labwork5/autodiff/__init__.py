"""A tiny automatic-differentiation library (Week 1 project)."""
from autodiff.dual import Dual, derivative
from autodiff.var import Var
from autodiff.functions import cos, exp, log, sin
from autodiff.gradient import gradient_forward, gradient_reverse, numerical_gradient

__all__ = [
    "Dual", "derivative", "Var",
    "exp", "log", "sin", "cos",
    "gradient_forward", "gradient_reverse", "numerical_gradient",
]
