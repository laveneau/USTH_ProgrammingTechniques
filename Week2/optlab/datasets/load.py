"""Reading the prepared datasets. numpy only, and outside `src/` on purpose.

The `.npz` files in `data/` are written once by the instructor with `prepare.py`, which
is the only thing that touches scikit-learn. The clone therefore contains everything the
week needs and no network access is required after it.
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a prepared dataset.

    Available: california, diabetes, breast_cancer, digits_3v8, iris_setosa,
    a9a, sparse_synthetic, outliers, curve_* and the NIST StRD sets.
    """
    path = DATA / f"{name}.npz"
    if not path.exists():
        available = sorted(p.stem for p in DATA.glob("*.npz"))
        raise FileNotFoundError(f"no dataset {name!r} in {DATA}; available: {available}")
    with np.load(path) as f:
        return f["X"], f["y"]


def standardize(X: np.ndarray) -> np.ndarray:
    """Centre and scale each column to unit variance.

    Day 1 measures the condition number of XᵀX/n before and after this, on raw
    California housing: conditioning is partly a property of the data, and a great deal
    of optimization difficulty is created by not doing this.
    """
    raise NotImplementedError("[DAY 1] lab 3")
