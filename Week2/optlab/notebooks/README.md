# Notebooks

One per day. These are where you *look* at what you built — every algorithm gets a plot,
and every plot gets an explanation.

| Notebook | Day | What to show |
|---|---|---|
| `01_losses_and_conditioning.ipynb` | 1 | likelihood → loss; level sets against κ; the finite-difference error U-curve |
| `02_descent.ipynb` | 2 | the zigzag on an ill-conditioned quadratic; ‖∇f‖ on a log scale; `w` as an 8×8 image |
| `03_stochastic.ipynb` | 3 | loss against epoch for several batch sizes; the noise floor; SGD against Adam |
| `04_newton.ipynb` | 4 | Newton's error on a log-log scale (the digits doubling); the double well; the ridge path |
| `05_least_squares.ipynb` | 5 | Gauss-Newton against Levenberg-Marquardt trajectories; λ and ρ over the iterations |
| `06_sparsity_and_benchmark.ipynb` | 6 | the soft-threshold; the lasso path; Huber against squared error; the benchmark profiles |

## Running them

Each notebook opens with a **readiness cell** that reports `ok` / `MISSING` / `BROKEN` for
every piece of `optlab` it is about to use. Run it first. `MISSING` means you have not
written that lab yet, not that anything is wrong.

The notebooks themselves need only `numpy` and `matplotlib`. scipy and scikit-learn are
used as *oracles* — to check your answer against somebody else's — and they live in the
optional `dev` extra, so install them with:

```
pip install -e ".[dev]"
```

That is allowed here because this directory is outside `src/`; importing any of them from
`src/optlab/` fails `tests/test_no_oracle_in_src.py`, which is the point of the rule.

As shipped, none of the six notebooks imports scipy or sklearn — every oracle in them is
either a closed-form answer or a second independent implementation, so `numpy` and
`matplotlib` are enough to run all of them end to end. Reach for the `dev` extra when
*you* want to check your own work against an outside implementation.

For every algorithm: **derive it, code it, check it against an oracle, then break it.**
The last step is not optional. A method you have only seen succeed is a method you do
not yet understand — force the divergence, and be able to say why it happened.
