# optlab — student repository (Week 2: Optimization)

You will spend the whole week in this one repository. Clone it once; nothing else is downloaded afterwards.

## Setup (first session)

```bash
make install      # pip install -e ".[dev]"
make check        # everything should run; most tests fail, that is expected
```

## The rule of the week

**Interfaces are given. Implementations are yours.**

Four files are **provided and must not be edited**:

| File | Contains |
|---|---|
| `src/optlab/interfaces.py` | every abstract base class you will implement |
| `src/optlab/results.py` | `OptimizeResult`, `StepEvent` |
| `src/optlab/errors.py` | `NotPositiveDefiniteError`, `LineSearchFailed` |
| `src/optlab/types.py` | the `Vec` / `Mat` / `Index` aliases |

Everything else under `src/optlab/` is a stub: a signature, a docstring stating the
contract, and `raise NotImplementedError`. Replace the `NotImplementedError` with
working code. Each stub is tagged with the day it is filled in, e.g. `[DAY 4]`.

## Imports allowed inside `src/optlab/`

**`numpy` only.** `scipy`, `scikit-learn` and `matplotlib` are *oracles*: they may be
imported by `tests/`, `benchmarks/` and `notebooks/` to check your work, never by the
package itself. `tests/test_no_oracle_in_src.py` parses the source tree and fails if
one appears — it runs from day 1 as part of `make check`.

## Your Week 1 autodiff module

Copy your Week 1 project into `src/optlab/autodiff/` (`dual.py` and `tensor.py`),
replacing the placeholders. From day 1 it is used as a **gradient oracle**: every
gradient you write by hand is checked against it as well as against finite differences.

## Daily commands

```bash
make test DAY=2   # the tests of days 1 and 2
make contracts    # the interface contract tests, over every implementation
make check        # tests + mypy --strict + ruff + the no-oracle guard
```

`make check` must be green before you leave. A red `mypy --strict` counts as a failure.

## What is graded

Hidden tests 40 % · code quality (`mypy`, `ruff`, no dead code, no oracle in `src/`) 20 % ·
extensibility challenge 15 % · `DESIGN.md` 10 % · benchmark note 15 %.

Keep `DESIGN.md` up to date as you go — it is where you justify your design decisions,
and it is the only place an edit to a provided file could ever be defended.
