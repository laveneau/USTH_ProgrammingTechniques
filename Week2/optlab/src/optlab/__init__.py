"""optlab — optimization by code.

Only `numpy` may be imported inside this package. `scipy` and `scikit-learn` are
oracles for tests and benchmarks; `tests/test_no_oracle_in_src.py` enforces this.
"""

from .errors import LineSearchFailed, NotPositiveDefiniteError, OptlabError
from .results import OptimizeResult, StepEvent

__all__ = [
    "LineSearchFailed",
    "NotPositiveDefiniteError",
    "OptlabError",
    "OptimizeResult",
    "StepEvent",
]
