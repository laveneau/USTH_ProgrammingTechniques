"""INSTRUCTOR SCRIPT - run once before the course, not by students.

Writes `data/*.npz` so that the clone is self-contained and the week needs no network.
This is the only file allowed to import scikit-learn for data (it is outside `src/`).

    python datasets/prepare.py

It downloads California housing (scikit-learn), a9a (LIBSVM) and three NIST StRD sets.
With the python.org build on macOS, `SSL: CERTIFICATE_VERIFY_FAILED` means Python has no
root certificates: run "Install Certificates.command" from its Applications folder, or
prefix the command with `SSL_CERT_FILE=/etc/ssl/cert.pem`.

Datasets, by the day that uses them:
  1    california, diabetes        regression as MLE; conditioning raw vs standardized
  1-2  breast_cancer               logistic as MLE; gradient descent
  2    digits_3v8                  w reshaped to an 8x8 image
  3    a9a (or a covertype subset) realistic-scale stochastic training
  4    iris_setosa, breast_cancer  Newton/IRLS; separability; Fisher information
  5    curve_*, NIST StRD          Gauss-Newton and Levenberg-Marquardt; certified values
  6    diabetes, sparse_synthetic, outliers   ridge vs lasso; selection; Huber

Every file holds `X` (n, d) and `y` (n,), which is all `load()` returns. Some carry extra
arrays for the instructor and the notebooks, read with `np.load` directly:
  - synthetic sets: `x_true` (curves) or `w_true` (regression), the generating parameters;
  - NIST sets: `start1`, `start2` (the two official starts, far and near) and
    `certified`, `certified_sd` (the certified estimates and their standard deviations).
Binary labels are stored as 0.0/1.0. Curve sets store t as the single column of X.
"""

import io
import ssl
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_svmlight_file,
)

DATA = Path(__file__).resolve().parent.parent / "data"

A9A_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
# Set to an int to ship a random subsample of a9a instead of all 32561 rows.
A9A_ROWS: int | None = None

NIST_URL = "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/{}.dat"
# Misra1a is an exponential rise (lower difficulty), Thurber a rational function (higher),
# MGH09 is the classic case where the far start defeats Gauss-Newton (higher).
NIST_SETS = ("Misra1a", "Thurber", "MGH09")

SEED = 2026


def save(name: str, X: np.ndarray, y: np.ndarray, **extra: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    np.savez_compressed(DATA / f"{name}.npz", X=X, y=y, **extra)
    print(f"  {name:<18} X{X.shape}")


def fetch(url: str) -> bytes:
    # Python >= 3.13 rejects certificates without a Subject Key Identifier, which is the
    # case of the LIBSVM server. Relax only that check; the certificate is still verified.
    context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    # NIST answers 403 to urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "optlab-prepare/1.0"})
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        return bytes(response.read())


# --------------------------------------------------------------------------- #
# Real datasets
# --------------------------------------------------------------------------- #


def prepare_sklearn() -> None:
    # Raw on purpose: day 1 measures the conditioning before and after standardizing.
    california = fetch_california_housing()
    save("california", california.data, california.target)

    diabetes = load_diabetes(scaled=False)
    save("diabetes", diabetes.data, diabetes.target)

    cancer = load_breast_cancer()
    save("breast_cancer", cancer.data, cancer.target)

    digits = load_digits()
    keep = np.isin(digits.target, (3, 8))
    save("digits_3v8", digits.data[keep], digits.target[keep] == 8)

    # Setosa against the rest is linearly separable: the logistic MLE does not exist,
    # and Newton shows it by letting ||w|| grow without bound.
    iris = load_iris()
    save("iris_setosa", iris.data, iris.target == 0)


def prepare_a9a(rng: np.random.Generator) -> None:
    X, y = load_svmlight_file(io.BytesIO(fetch(A9A_URL)), n_features=123)
    X = X.toarray()
    if A9A_ROWS is not None:
        keep = rng.choice(X.shape[0], size=A9A_ROWS, replace=False)
        X, y = X[keep], y[keep]
    save("a9a", X, y > 0)


def parse_nist(text: str) -> dict[str, np.ndarray]:
    """Read a NIST StRD nonlinear-regression .dat file.

    The parameter lines read `b1 = start1 start2 certified certified_sd`, and the data
    are the rows after the last line starting with `Data:`, in the order y, x.
    """
    lines = text.splitlines()
    params = []
    for line in lines:
        head, _, tail = line.partition("=")
        if head.strip().startswith("b") and head.strip()[1:].isdigit():
            params.append([float(v) for v in tail.split()[:4]])
    start = max(i for i, line in enumerate(lines) if line.strip().startswith("Data:"))
    body = [line for line in lines[start + 1 :] if line.strip()]
    rows = np.array([[float(v) for v in line.split()] for line in body])
    table = np.array(params)
    return {
        "X": rows[:, 1],
        "y": rows[:, 0],
        "start1": table[:, 0],
        "start2": table[:, 1],
        "certified": table[:, 2],
        "certified_sd": table[:, 3],
    }


def prepare_nist() -> None:
    for name in NIST_SETS:
        nist = parse_nist(fetch(NIST_URL.format(name)).decode("ascii", errors="replace"))
        X, y = nist.pop("X"), nist.pop("y")
        save(f"nist_{name.lower()}", X, y, **nist)
    # No `curve_nist`: benchmarks/run.py would fit it with ExpDecay from (1, 0.2, 0), and
    # on Misra1a (x up to 790, b2 ~ 5.5e-4) exp(-0.2*x) underflows, so every method fits a
    # constant. The benchmark keeps its synthetic exponential decay instead.


# --------------------------------------------------------------------------- #
# Synthetic datasets
# --------------------------------------------------------------------------- #


def prepare_curves(rng: np.random.Generator) -> None:
    """One set per model of problems/curve_fitting.py, with the truth alongside."""
    t = np.linspace(0.0, 4.0, 60)
    truth = np.array([3.0, 1.4, 0.5])
    y = truth[0] * np.exp(-truth[1] * t) + truth[2] + rng.normal(scale=0.02, size=t.size)
    save("curve_expdecay", t, y, x_true=truth)

    truth = np.array([2.0, 1.5, 0.4])
    y = truth[0] * np.exp(-((t - truth[1]) ** 2) / (2.0 * truth[2] ** 2))
    save("curve_gaussian", t, y + rng.normal(scale=0.03, size=t.size), x_true=truth)

    t = np.linspace(0.0, 2.0 * np.pi, 80)
    truth = np.array([1.5, 2.0, 0.5])
    y = truth[0] * np.sin(truth[1] * t + truth[2])
    save("curve_sinusoid", t, y + rng.normal(scale=0.05, size=t.size), x_true=truth)


def prepare_sparse(rng: np.random.Generator) -> None:
    """More features than the signal needs: 5 of 50 matter, n = 100."""
    n, d = 100, 50
    X = rng.normal(size=(n, d))
    w_true = np.zeros(d)
    support = rng.choice(d, size=5, replace=False)
    w_true[support] = rng.choice((-1.0, 1.0), size=5) * rng.uniform(1.0, 3.0, size=5)
    save("sparse_synthetic", X, X @ w_true + rng.normal(scale=0.5, size=n), w_true=w_true)


def prepare_outliers(rng: np.random.Generator) -> None:
    """y = 2x + noise through the origin, with 10% of the points pushed far upward."""
    n = 50
    x = rng.uniform(0.0, 10.0, size=n)
    w_true = np.array([2.0])
    y = w_true[0] * x + rng.normal(scale=0.5, size=n)
    bad = rng.choice(n, size=n // 10, replace=False)
    y[bad] += rng.uniform(20.0, 40.0, size=bad.size)
    save("outliers", x, y, w_true=w_true)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"writing into {DATA}")
    prepare_sklearn()
    prepare_a9a(rng)
    prepare_nist()
    prepare_curves(rng)
    prepare_sparse(rng)
    prepare_outliers(rng)


if __name__ == "__main__":
    main()
