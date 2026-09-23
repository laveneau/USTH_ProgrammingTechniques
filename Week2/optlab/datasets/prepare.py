"""INSTRUCTOR SCRIPT — run once before the course, not by students.

Writes `data/*.npz` so that the clone is self-contained and the week needs no network.
This is the only file allowed to import scikit-learn for data (it is outside `src/`).

    python datasets/prepare.py

Datasets, by the day that uses them:
  1    california, diabetes        regression as MLE; conditioning raw vs standardized
  1-2  breast_cancer               logistic as MLE; gradient descent
  2    digits_3v8                  w reshaped to an 8x8 image
  3    a9a (or a covertype subset) realistic-scale stochastic training
  4    iris_setosa, breast_cancer  Newton/IRLS; separability; Fisher information
  5    curve_*, NIST StRD          Gauss-Newton and Levenberg-Marquardt; certified values
  6    diabetes, sparse_synthetic, outliers   ridge vs lasso; selection; Huber
"""

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    raise NotImplementedError(
        "Instructor: fetch each dataset with scikit-learn/LIBSVM and np.savez it into data/. "
        "Ship a subsample of a9a if the full set is too large for the room."
    )


if __name__ == "__main__":
    main()
