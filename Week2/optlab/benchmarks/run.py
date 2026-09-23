"""[DAY 6] Run every optimizer over every problem and write a CSV for notebook 06.

Outside `src/`, so scipy and scikit-learn are available here as oracles.

Record per (problem, method): iterations or epochs, function and gradient evaluations,
wall-clock, final loss, and the gap to the oracle's minimizer. The deliverable is not
the CSV but the one-page answer to "which optimizer for which problem?" — and the
honest version of that answer names the cases where the fancy method loses.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "results.csv"

PROBLEMS = [
    "quadratic_illconditioned",
    "rosenbrock",
    "linear_california_raw",
    "logistic_breast_cancer",
    "logistic_digits_3v8",
    "curvefit_nist",
    "lasso_diabetes",
    "logistic_a9a_large",
]

METHODS = [
    "gradient_descent_armijo",
    "momentum",
    "sgd",
    "adam",
    "newton",
    "ridge_newton",
    "gauss_newton",
    "levenberg_marquardt",
    "lasso_ista",
]


def main() -> None:
    raise NotImplementedError("[DAY 6] lab 3")


if __name__ == "__main__":
    main()
