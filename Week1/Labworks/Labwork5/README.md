# autodiff — Week 1 project: automatic differentiation

In this project you build a small library that computes **exact derivatives of Python
code**, in the two modes used by every machine-learning framework:

- **forward mode** with *dual numbers* (`Dual`);
- **reverse mode**, a.k.a. **backpropagation** (`Var`).

It closes Week 1 — it uses classes, typing, tests and SOLID — and it is the tool
Week 2 (optimization) relies on to check every gradient.

Read `Lecture5-autodiff.ipynb` first.

## Setup

```bash
git clone <this repository> autodiff-lab
cd autodiff-lab
python -m pip install mypy          # the tests only need the standard library
```

Python ≥ 3.10.

## Layout

```
autodiff/
├── __init__.py      # public API                                   (provided)
├── dual.py          # Part 1 — forward mode                         TO COMPLETE
├── var.py           # Part 2 — reverse mode                         TO COMPLETE
├── gradient.py      # Part 3 — gradients of f: ℝⁿ → ℝ               TO COMPLETE
└── functions.py     # exp, log, sin, cos for float/Dual/Var         (provided)
tests/               # unit tests                                    (provided, do not modify)
notebooks/
└── autodiff-plots.ipynb   # plots, once your code passes the tests
```

## What to do

Replace every `raise NotImplementedError("TODO: ...")`. In each class, `__add__` is
given as a worked example: follow the same pattern.

| Part | File | Command | Suggested time |
|---|---|---|---|
| 1 — dual numbers | `autodiff/dual.py` | `make part1` | 45 min |
| 2 — backpropagation | `autodiff/var.py` | `make part2` | 95 min |
| 3 — gradients | `autodiff/gradient.py` | `make part3` | 20 min |

Work **test by test** (red → green): read a failing test, make it pass, commit.

Without `make`:

```bash
python -m unittest -v tests/test_dual.py        # one part
python -m unittest discover -s tests            # everything
mypy --strict autodiff                          # typing
```

The suite is written with `unittest`, so nothing has to be installed. If you have
`pytest` (the end of Lecture 3, and the runner used in Week 2), it runs these very same
tests unchanged and gives you nicer failure messages:

```bash
python -m pytest                                # everything
python -m pytest tests/test_dual.py -x          # one part, stop at the first failure
```

## Done when

- `make check` is green: all tests pass **and** `mypy --strict autodiff` reports no issue;
- the plots notebook runs from top to bottom;
- your history shows small commits (one per group of tests).

## Rules

- Do not modify `tests/`, `autodiff/functions.py` nor `numerical_gradient`.
- Only the standard library (`math`) in `autodiff/`.
- Keep every method short: if you copy-paste, look for a helper (DRY).

## Hints

- An unsupported operand must `return NotImplemented` (never raise): Python then
  tries the reflected method (`__radd__`, …) and raises the proper `TypeError`.
- `a - b` and `a / b` are not commutative: check `__rsub__` and `__rtruediv__`.
- In reverse mode, a node used twice receives two contributions: **accumulate**.
- `topological_order` must be iterative (explicit stack), not recursive.

## Bonus

1. Add `tanh` (to `Dual`, `Var` and `functions.py`) and train a tiny neuron.
2. Add `__pow__` with a *variable* exponent: `x ** y = exp(y·log x)`.
3. Compare with the complex-step method (see the lecture).
