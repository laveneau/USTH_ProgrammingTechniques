# optlab - student repository (Week 2: Optimization)

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
one appears - it runs from day 1 as part of `make check`.

## The `autodiff` module

`src/optlab/autodiff/` is **provided** - do not edit it, and you do not need to supply it.
It is the Week 1 project generalized from single numbers to numpy arrays, which is what the
rest of `optlab` speaks. Nothing in `src/optlab/` imports it; it exists so that `tests/`
can check a gradient against a computation that is **exact**, where finite differences stop
at about six correct digits.

    from optlab.autodiff import autodiff_gradient, jacobian_forward, exp, log

`autodiff_gradient` is reverse mode - one sweep for a whole gradient, whatever `p` is, and
what day 1 uses. `jacobian_forward` is forward mode - one sweep per parameter, each filling
one column, which is the right shape for day 5's Jacobians.

Your own Week 1 module works on single numbers. Keep it where it is: day 1's labwork asks
you to read `tensor.py` against it and to explain the difference, but Week 2 never imports
it, so an unfinished Week 1 project blocks nothing here.

## Daily commands

```bash
make test DAY=2   # the tests of days 1 and 2
make contracts    # the interface contract tests, over every implementation
make check        # tests + mypy --strict + ruff + the no-oracle guard
```

`make check` must be green before you leave. A red `mypy --strict` counts as a failure.

## What is graded

Hidden tests 40 % - code quality (`mypy`, `ruff`, no dead code, no oracle in `src/`) 20 % *
extensibility challenge 15 % - `DESIGN.md` 10 % - benchmark note 15 %.

Keep `DESIGN.md` up to date as you go - it is where you justify your design decisions,
and it is the only place an edit to a provided file could ever be defended.
