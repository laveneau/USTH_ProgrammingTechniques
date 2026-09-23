# Optimization by Code — Week 2 of a 50 h course

**MSc Data Science · Week 2 · 25 h · 6 days × ≈ 4 h 15**

---

## 0. Context

This document covers **Week 2 (25 h): optimization**. For the two-week course structure, the Week 1 syllabus, and how the weeks connect, see the [root `README.md`](../README.md); the Week 1 plan itself is `Week1/course_plan_week1_EN.md`.

**Prerequisites actually held after Week 1** (read from the notebooks, so the plan matches reality):
| Tool | Status coming into the project | Consequence |
|---|---|---|
| Classes, inheritance (incl. multiple), `@property`, `__iter__`, `__call__` | known | interfaces use multiple inheritance of ABCs, as in L1/L4 |
| Interfaces via **`ABC` / `abstractmethod`** | known (L4, Labwork 3–4) | **all provided interfaces are `ABC`s**, not `Protocol` |
| **Duck typing** (concept) | known (L2) | `Protocol` is mentioned once as the *structural* twin of the ABCs; not required |
| `mypy --strict`, `unittest`, `pytest` | known | `make check` and the test suite build directly on these. Week 1 ran on `unittest`; `pytest` -- fixtures, `parametrize`, markers, `approx` -- was introduced at the end of Week 1's Lecture 3 and is the runner here |
| **Operator overloading** (`__add__`, `__mul__`, …) | known | **introduced in the Week 1 project** (autodiff `Dual`/`Tensor`) — a natural first use |
| **`@dataclass`** | known | introduced in the Week 1 project (for `Dual`); reused here for `OptimizeResult`, `StepEvent` |
| **numpy** | barely seen (one Week-1 lab) | **short numpy ramp on Week 2 · Day 1** |

Week-2 arc: **losses (MLE) → gradient descent → stochastic training → second-order & statistics → nonlinear least squares (GN/LM) → regularization & sparsity.** Autodiff (Week 1) underlies all of it.

---

## A. What arrives from Week 1: the `autodiff` module

Week 1 closes with an **automatic differentiation** project — the bridge into this week. It is specified in full in `Week1/course_plan_week1_EN.md` (section "Project"); only what Week 2 depends on is restated here.

**What students bring in.** A working `autodiff` package, carried into the `optlab` repository as `src/optlab/autodiff/`:

| Module | Contents | Mode |
|---|---|---|
| `autodiff/dual.py` | `Dual(val, der)` with full operator overloading, `exp`, `log`, `derivative(f, x)`, `gradient_forward(f, x)` | forward — `n` sweeps for a gradient |
| `autodiff/tensor.py` | `Tensor` with `+ − * / ** @ sum T exp log` and `backward()` | reverse (= backprop) — `O(1)` sweeps for a scalar gradient |
| — | `autodiff_gradient(f)` | the entry point used here |

**How Week 2 uses it.** As a **gradient oracle**: from Day 1, every hand-written `gradient` is validated against `autodiff_gradient` alongside the finite-difference `check_gradient`. Two independent oracles catch different mistakes — finite differences catch sign and scale errors, autodiff catches them exactly to machine precision.

**Two skills it establishes**, both first met there and assumed here: **arithmetic operator overloading** (the dunders `__add__`, `__mul__`, …) and **`@dataclass`** — the latter reused immediately for the provided `OptimizeResult` and `StepEvent`.

---

# WEEK 2 — OPTIMIZATION

## 1. Overview

### Goal
Teach the foundations of optimization by programming them, using ML model fitting as the running case study. Students extend the library `optlab` and use it to fit real models by maximum likelihood, covering the techniques that dominate practice: **first-order** descent, **stochastic** training, **second-order** methods, **nonlinear least squares**, and **regularized / sparse** estimation.

Spirit (Knuth): you understand an algorithm once you have written it, broken it, and fixed it. Mathematics **predicts** what the code will do; the code confirms or refutes.

### The double stake, made explicit
Each day serves both goals: **programming** (extend a clean, typed, tested, SOLID API by implementing provided interfaces) and **data science** (understand *why* a model is fit the way it is — the loss as a likelihood, conditioning as a data property, stochastic gradients as the answer to "too much data", regularization as a prior, GN/LM as the workhorse of curve fitting).

### Through-line
`optlab`, **pure SOLID**: interfaces provided on day 1, implementations written day by day. The repo is cloned once; the same code grows. The Week-1 `autodiff` module is carried into the repo and used as a gradient oracle.

### Ground rules
| Component | Status |
|---|---|
| `numpy` | **Floor**: arrays, `@`, dense linear algebra. Only import allowed in `src/optlab/` |
| `optlab.autodiff` | **Available** (built Week 1): used as a gradient oracle in tests |
| `scipy`, `scikit-learn` | **Oracles**: tests/benchmarks only, never in the package |
| Everything else | **Written by the students**: losses, Cholesky, all optimizers, proximal operators, GN/LM |

A provided `ast` test fails if `scipy`/`sklearn` appears in `src/optlab`. Runs from day 1 in `make check`.

### Learning outcomes
1. Read a model as a loss, a loss as a negative log-likelihood, regularization as a prior.
2. Write optimality conditions; predict difficulty (convexity, conditioning).
3. Choose and implement the right optimizer: batch vs stochastic, first- vs second-order, smooth vs proximal, general vs least-squares.
4. Design and extend an API in pure SOLID: small stable interfaces, interchangeable implementations, injected dependencies, contract tests.

---

## 2. Pedagogical principles

### Daily rhythm (255 min)

| Block | min |
|---|---|
| Lecture | **40** (hard cap 45) |
| Break | 10 |
| Labwork | **165** |
| Debrief | 20 |
| Slack | 20 |

The lecture cap is deliberately tight and the labwork block deliberately long: this week
is learned at the keyboard. Each lecture notebook opens with a **pacing table** giving a
per-section budget and marking the sections to cut first when running behind.

**Every lecture is illustrated.** The six notebooks carry 28 figures between them — 4 to 6
each — produced by the code cell immediately above each one, so they stay honest: every
number quoted in a caption is a number the notebook computed. The pacing tables already
include the time to show and discuss them, and each table names the figure that can be
dropped first. **Run the whole notebook once before the session**; matplotlib is the only
added dependency.

| Lecture | figures | core / total min |
|---|---|---|
| 1 — Losses and likelihood | 5 | 44 / 48 · §1 can be set as pre-reading |
| 2 — Gradient descent and line search | 5 | 40 / 44 |
| 3 — Stochastic optimization | 4 | 35 / 40 |
| 4 — Second-order methods | 4 | 41 / 44 |
| 5 — Nonlinear least squares | 5 | 39 / 41 |
| 6 — Sparsity and robustness | 5 | 37 / 41 |

Lecture 1 is the one at risk of overrunning, and deliberately so: its §1 is the calculus
recall (gradient, Hessian, saddle points, Taylor, convexity) that everything else stands
on. With a group already fluent in multivariate calculus, set §1 as pre-reading and open
the session at §2, which brings it back to 32 min.

Each day's labwork is a notebook — `Labworks/LabworkN/LabworkN.ipynb` — following the
Week 1 convention: a short statement per exercise, the file to open, and a cell that runs
the relevant tests. The code itself is written in the `optlab` package, not in the
notebook, because the architecture *is* the subject this week.

### The loop for every algorithm
1. **Derive** at the board. 2. **Code** against provided tests. 3. **Check** against an oracle (autodiff, scipy, scikit-learn) in tests only. 4. **Observe and explain**: plot convergence, recover the rate, then **force a failure** and say why.

### Illustrate every notion three times
(1) a hand-worked 3–5 line example (small numbers, becomes a test); (2) a numerical example in the lab; (3) a visual in the notebook. Problems recur across days (gallery, §6).

### Density guardrails
One headline theme per day; ≤ 3 new interfaces per day in depth; **lecture ≤ 45 min**; every lab ends on something that runs and passes.

### Repository: clone once, then no network
Everything is in the clone (architecture, six days of skeletons, all tests, notebooks, data, and the Week-1 `autodiff` module). No `git pull` afterwards. Single `main`, no per-day tags. `make test DAY=n` runs days 1–n; `make check` adds `mypy --strict`, `ruff`, "no scipy in `src/`". Visible tests provided; hidden tests for grading. Environment prepared in advance (Docker / conda / wheels). A stuck student never blocks the rest (interfaces provided; pairs; one reference module on request).

### Real data (prepared once by the instructor)
`datasets/prepare.py` (before the course) writes `data/*.npz`; `datasets/load.py` (numpy only, outside `src/`) reads them.

| Day | Dataset | Source | Illustrates |
|---|---|---|---|
| 1 | *California housing*, *diabetes* | scikit-learn | regression as MLE; conditioning raw vs standardized |
| 1–2 | *breast cancer* (569×30) | scikit-learn | logistic as MLE; gradient descent |
| 2 | *digits* (3 vs 8) | scikit-learn | `w` reshaped to an 8×8 image |
| 3 | *a9a* (LIBSVM) or *covertype* subset | LIBSVM/HF | realistic-scale stochastic training |
| 4 | *iris* (setosa vs rest), *breast cancer* | scikit-learn | Newton/IRLS; separability; Fisher information |
| 5 | Curve-fit sets; **NIST StRD** (*Misra1a*, *Thurber*, *MGH09*) | synthetic / NIST | GN/LM; certified values; easy/hard starts |
| 6 | *diabetes*, synthetic sparse, outlier-contaminated | scikit-learn / synthetic | ridge vs lasso; feature selection; Huber |

We do optimization, not "ML": accuracy is a sanity check; the real check is agreement with scikit-learn's minimizer.

---

## 3. SOLID architecture: interfaces provided, implementations to write

All interfaces are shipped whole in `interfaces.py` as **abstract base classes** (`ABC` + `@abstractmethod`), exactly the mechanism Week 1 uses for SOLID (Lecture 4, Labwork 3–4), and following the two conventions stated there: each is named with a leading capital `I`, and each is **pure** — every method abstract, no implementation and no state, so every line of behaviour behind a contract is written by the student. Students write **implementations** that inherit them, and never edit an interface. An implementation may inherit **several** ABCs (e.g. `GLMLoss(IObjective, ITwiceDifferentiable, IBatchObjective)`) — a direct use of the multiple inheritance seen in Lecture 1. Structurally these are the "duck-typed" contracts of Lecture 2; `Protocol` is the structural-typing twin, mentioned once and not required. The Week-1 `autodiff` module is present and used as a gradient oracle.

### The interfaces (abstract base classes)
| Interface | Methods | Implementations to write | Day | Depended on by |
|---|---|---|---|---|
| `IObjective` | `value(x)`, `gradient(x)` | `Quadratic`, `Rosenbrock`, `GLMLoss` | 1 | optimizers, line search, gradient check |
| `IPointwiseLoss` | `value(z,y)`, `d1(z,y)`, `d2(z,y)` | `SquaredError`, `LogisticNLL`, `Huber`, `PoissonNLL` | 1 (Huber/Poisson D6) | `GLMLoss` |
| `IBatchObjective` | `n_samples`, `batch_gradient(x, idx)` | `GLMLoss` | 3 | `SGD`, `Adam` |
| `ITwiceDifferentiable` | `hessian(x)` | `Quadratic`, `Rosenbrock`, `GLMLoss` | 4 | `NewtonDirection` |
| `ILinearSolver` | `solve(A, b) -> Vec` | `CholeskySolver` | 4 | `NewtonDirection`, `GaussNewton`, `LevenbergMarquardt` |
| `ILeastSquaresProblem` | `residuals(x)`, `jacobian(x)` | curve-fit problems | 5 | `GaussNewton`, `LevenbergMarquardt` |
| `IRegularizer` | `value(w)`, `gradient(w)`, `prox(w, t)` | `NoRegularizer`, `L2` (D4), `L1`, `ElasticNet` (D6) | 4–6 | `RegularizedObjective`, `ProximalGradient` |
| `ILineSearch` | `step(objective, x, g, d)` | `FixedStep`, `Armijo` | 2 | `DescentOptimizer` |
| `IDirectionRule` | `direction(objective, x, g)` | `SteepestDescent`, `HeavyBall`, `NewtonDirection` | 2–4 | `DescentOptimizer` |
| `IStoppingCriterion` | `should_stop(event)` | `GradientNormBelow`, `MaxIterations`, `AnyOf` | 2 | all optimizers |
| `IObserver` | `on_step(event)` | `History` | 2 | all optimizers |
| `IOptimizer` | `minimize(objective, x0) -> OptimizeResult` | `DescentOptimizer` (D2), `SGD`/`Adam` (D3), `GaussNewton`/`LevenbergMarquardt` (D5), `ProximalGradient` (D6) | 2–6 | notebooks, benchmark |

Provided (not written): `OptimizeResult`, `StepEvent` (frozen dataclasses — the `@dataclass` first met in the Week 1 project), `NotPositiveDefiniteError`, `LineSearchFailed`.

### The wiring: everything is composition
```
DescentOptimizer(direction, line_search, stop, observers)      # deterministic loop, once (D2)

Gradient descent = SteepestDescent            + Armijo
Momentum         = HeavyBall                  + FixedStep
Newton           = NewtonDirection(CholeskySolver) + Armijo                (D4)
Ridge (smooth)   = any of the above, on RegularizedObjective(GLMLoss, L2)  (D4)
SGD / Adam       = own Optimizers over an IBatchObjective                   (D3)
Gauss–Newton     = GaussNewton(CholeskySolver)  on an ILeastSquaresProblem  (D5)
Levenberg–Marq.  = LevenbergMarquardt(CholeskySolver) (JᵀJ + λI)            (D5)
Lasso            = ProximalGradient(smooth=GLMLoss, reg=L1)                 (D6)
```
Showcase OCP moments: **ridge** = any existing optimizer on `RegularizedObjective(loss, L2)`; **Newton-CG-free GN** = the same `CholeskySolver` reused on `JᵀJ`; **lasso** = a new `ProximalGradient` reusing `GLMLoss`.

### SOLID, principle by principle
| Principle | In the course code |
|---|---|
| **S** | `GLMLoss` = data-fit loss; `IRegularizer` = penalty (separated); `CholeskySolver` = factorization; `GaussNewton` = the outer iteration, not the solve |
| **O** | SGD, Adam, Newton, ridge, GN, LM, lasso, Poisson, Huber are added without editing existing classes |
| **L** | contract tests over all implementations: `IObjective`, `IPointwiseLoss`, `ILinearSolver`, `IRegularizer` |
| **I** | descent needs only `IObjective`; SGD only `IBatchObjective`; Newton adds `ITwiceDifferentiable`; GN/LM depend on `ILeastSquaresProblem` (not `IObjective`); proximal needs only `IRegularizer.prox` |
| **D** | optimizers receive direction / line search / stopping / observers / rng / **solver** by constructor; `GLMLoss` receives its `IPointwiseLoss` |

### KISS — deliberately not built
A home-grown `Vector`; `DidNotConverge`; `requires_grad`/op-registry (autodiff kept simple, Week 1); sparse matrices / CG / PCG (out this revision). Generic `Num2` lives in `labs/`.

---

## 4. Repository tree
"D" marks the Week-2 day a file is filled in; `autodiff/` arrives from Week 1.
```
optlab/
├── pyproject.toml · Makefile · DESIGN.md
├── src/optlab/
│   ├── types.py · interfaces.py(PROVIDED: ABCs) · errors.py(PROVIDED) · results.py(PROVIDED)
│   ├── numerics/gradcheck.py          # D1 numerical_gradient/jacobian, check_gradient
│   ├── autodiff/{dual.py,tensor.py}   # from Week 1 (oracle)
│   ├── losses.py                      # D1 SquaredError, LogisticNLL · D6 Huber, PoissonNLL
│   ├── problems/
│   │   ├── glm.py                     # D1 GLMLoss (value, gradient) · D4 hessian
│   │   ├── quadratic.py · rosenbrock.py  # D1 · D4 hessian
│   │   └── curve_fitting.py           # D5 ILeastSquaresProblem instances
│   ├── regularizers.py                # D4 L2 · D6 L1, ElasticNet, NoRegularizer
│   ├── objective_ops.py               # D4 RegularizedObjective (adapter)
│   ├── linalg/cholesky.py             # D4 CholeskySolver
│   ├── linesearch.py · stopping.py · observers.py   # D2
│   └── optimizers/
│       ├── descent.py                 # D2 DescentOptimizer + gradient_descent(), momentum(), newton()
│       ├── directions.py              # D2 SteepestDescent, HeavyBall · D4 NewtonDirection
│       ├── stochastic.py              # D3 SGD, Adam
│       ├── least_squares.py           # D5 GaussNewton, LevenbergMarquardt
│       └── proximal.py                # D6 ProximalGradient (ISTA/FISTA)
├── tests/{unit,contracts}/ · labs/algebra2.py · datasets/ · data/ · benchmarks/run.py · notebooks/
```

---

## 5. Week 2 · day-by-day

---

### Day 1 — Losses, likelihood, and the optimization problem
**Headline.** A model is a loss; a loss is a negative log-likelihood. Set up the problem, verify a gradient (finite differences **and** the Week-1 autodiff), implement the first losses.

**Deliverable.** `numerics/gradcheck.py`; `losses.py` (`SquaredError`, `LogisticNLL`); `GLMLoss.value/gradient`; `Quadratic`, `Rosenbrock`.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | Lecture |
| 0:55 | 25 | Setup + guided tour of the architecture (the ABCs) |
| 1:20 | 25 | Lab 0: **numpy ramp** (arrays, broadcasting, `@`, `np.linalg`) |
| 1:45 | 35 | Lab 1: gradient check (FD + autodiff oracle) |
| 2:20 | 55 | Lab 2: `IPointwiseLoss` + `GLMLoss` |
| 3:15 | 20 | Lab 3: conditioning & the data |
| 3:35 | 20 | Debrief |

**Lecture.** `min f(x)`: stationarity, `∇²f≻0`, convexity ⇔ `∇²f⪰0`. **Loss = likelihood**: Gaussian ⇒ squared error; Bernoulli ⇒ logistic/cross-entropy; Poisson ⇒ `eᶻ−yz` (D6). GLM family `L(w)=(1/n)Σφ(zᵢ,yᵢ)`. **Regularization = prior (MAP)**: `½λ‖w‖²` ↔ Gaussian, `λ‖w‖₁` ↔ Laplace (foreshadows D4/D6). **Conditioning** is partly a data property (feature scales). Finite differences (forward/central, optimal `h`).

**Worked examples.** Squared error by hand (`X=[[1],[2]], y=(2,4), w=0` → value `5`, grad `−5`); logistic by hand (`y=(0,1)` → `log 2`, `−0.25`); overflow trap (`softplus(1000)=1000`); conditioning from scale (`κ≈10⁶` → tens after standardizing).

**Lab 0 — numpy ramp (25 min).** A guided, test-backed warm-up, since numpy was barely seen in Week 1: vectors/matrices, `reshape`, broadcasting rules, `@` vs `*`, reductions (`sum`, `mean`, `axis`), `np.linalg.solve`/`eigvalsh`. Small provided tests (e.g. "compute `Xᵀ(Xw−y)` without a Python loop") calibrate the level. Everything afterwards assumes this fluency.

**Lab 1 — Gradient check.** `numerical_gradient`, `check_gradient`; also compare to the Week-1 `autodiff_gradient` as a second oracle. Tests incl. a deliberately wrong gradient; error-vs-`h` U-curve.

**Lab 2 — Losses and `GLMLoss`.** `SquaredError`, `LogisticNLL` (`IPointwiseLoss`); `GLMLoss(X, y, pointwise)` (value = mean of φ; gradient = `Xᵀ·d1/n`); `linear_regression`, `logistic_regression` assemblies; `Quadratic`, `Rosenbrock`. **One class for all GLMs** (DRY/SRP), receiving the pointwise loss (DIP); **no `λ` yet** (regularizer is separate, D4). Tests: pointwise derivatives vs numerical; `GLMLoss.gradient` via `check_gradient` **and** autodiff; hand values; `IObjective` contract test over all four. Oracle: ridge closed form; logistic vs scikit-learn later.

**Lab 3 — Conditioning & data.** On raw *California housing*: build `A=XᵀX/n`; compute `κ` before/after standardization (oracle); see the level sets stretch. Sets up D2/D4.

**SOLID / NB / Plan B / Q.** S: loss vs future regularizer vs problem. I: `IObjective` = two methods. NB: likelihood→loss picture; level sets vs κ; FD curve. Plan B: ship `value`, write `gradient`. Q: why is squared error the Gaussian MLE? why does scale change κ?

---

### Day 2 — Gradient descent and line search
**Headline.** The simplest optimizer, done right, and the one loop every later optimizer plugs into.

**Deliverable.** `DescentOptimizer`; `SteepestDescent`, `HeavyBall`; `FixedStep`, `Armijo`; stopping criteria; `History`.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | Lecture |
| 0:55 | 55 | Lab 1: the `DescentOptimizer` loop |
| 1:50 | 55 | Lab 2: Armijo backtracking |
| 2:45 | 50 | Lab 3: momentum & conditioning |
| 3:35 | 20 | Debrief |

**Lecture.** `x⁺=x−α∇f`; descent lemma (`α≤1/L`); rate `(κ−1)/(κ+1)` on a quadratic; the zigzag. Armijo `f(x+αd)≤f(x)+c₁αgᵀd`. Momentum → `(√κ−1)/(√κ+1)`.

**Worked examples.** `f=½(x₁²+100x₂²)`: GD ≈ **690** steps vs heavy ball ≈ **70** vs Newton **1** (D4). Armijo by hand (`f=x²`, reject `α=1`, accept `0.5`).

**Lab 1 — The one loop.** `DescentOptimizer(direction, line_search, stop, observers)`; `SteepestDescent`; `FixedStep`; `GradientNormBelow/MaxIterations/AnyOf`; `History` (IObserver). **Written once, never again**: Newton (D4) will be just another direction. Tests: convergence on a well-conditioned quadratic; empirical rate within ±10 %; `converged=False` at `max_iter`; one event per step (SRP).

**Lab 2 — Armijo.** Injected in place of `FixedStep`, **no loop edit** (OCP); `LineSearchFailed` caught → `converged=False`. Compare on `κ=100` and on logistic (*breast cancer*). Tests: hand example; `ILineSearch` contract test.

**Lab 3 — Momentum & conditioning.** `HeavyBall` (stateful `IDirectionRule`). Compare GD / GD+Armijo / heavy ball; recover rates; on *digits* reshape `w` to 8×8. Force divergence with too-large a step; explain via `1/L`.

**SOLID / NB / Plan B / Q.** S: loop/step/stopping/recording separate. O: momentum/Armijo add no line. NB: zigzag; log `‖∇f‖`; `w`-image. Plan B: ship `Armijo` skeleton. Q: why is `1/L` safe? why does momentum help?

---

### Day 3 — Stochastic optimization (how modern ML trains)
**Headline.** When `n` is large, use a *sample* of the gradient. SGD, schedules, momentum, Adam.

**Deliverable.** `IBatchObjective` on `GLMLoss`; `SGD`, `Adam`.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | Lecture |
| 0:55 | 45 | Lab 1: mini-batch gradient |
| 1:40 | 55 | Lab 2: `SGD` (+ schedules, momentum) |
| 2:35 | 45 | Lab 3: `Adam` |
| 3:20 | 15 | Lab 4: batch-size / noise study |
| 3:35 | 20 | Debrief |

**Lecture.** Finite sum `f=(1/n)Σfᵢ`; full gradient `O(n)`, mini-batch `O(b)` **unbiased**, variance `∝1/b`. Constant step → noise floor `∝α`; decaying step (Robbins–Monro) → converges. Momentum smooths noise. **Adam**: per-coordinate step from moments `m,v` + bias correction (`β₁=0.9,β₂=0.999,ε=1e-8`); it *auto-standardizes* the per-coordinate scale — the optimization-side answer to conditioning. Epoch, shuffling, seeds.

**Worked examples.** Variance `∝1/b`; `b=n` ⇒ SGD = gradient descent (a test); Adam's first step ≈ `lr·sign(g)`; halving `α` halves the noise floor.

**Lab 1 — Mini-batch gradient.** `GLMLoss.n_samples`, `batch_gradient(w, idx)` (DRY: `gradient` = batch over all indices). Tests: batch-over-all = `gradient`; unbiasedness across random batches.

**Lab 2 — SGD.** Injected `np.random.Generator` (DIP, reproducibility); constant vs decaying step; optional momentum. Tests: `batch_size=n` ⇒ gradient descent; seeded reproducibility; decaying-step convergence.

**Lab 3 — Adam.** `m, v`, bias correction. Tests: first step ≈ `lr·sign(g)`; converges on logistic; on the ill-scaled (1 vs 1000) problem, far fewer epochs than SGD. Oracle: final loss vs scikit-learn `SGDClassifier`.

**Lab 4 — Batch/noise study.** On *a9a*/*covertype*: loss vs **epoch** for `b∈{1,32,256,n}` and SGD vs Adam; wall-clock vs epochs; noise floor. Mini-conclusion feeds the D6 benchmark note.

**SOLID / NB / Plan B / Q.** O: SGD/Adam are new `IOptimizer`s; the D2 loop and `GLMLoss` untouched. I: depend only on `IBatchObjective`. Plan B: ship the epoch/shuffle loop; write the updates. Q: why unbiased? why does a constant step not reach the exact optimum? what does Adam adapt to?

---

### Day 4 — Second-order methods and the statistics of the loss
**Headline.** Use curvature: Newton converges in a handful of steps; its Hessian is the Fisher information; IRLS is how GLMs are fit in statistics. Solve with Cholesky, never an inverse. Introduce the **regularizer** as a separate object (ridge for free).

**Deliverable.** `GLMLoss.hessian`; `CholeskySolver`; `NewtonDirection`; `L2` + `RegularizedObjective`.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | Lecture |
| 0:55 | 55 | Lab 1: Cholesky |
| 1:50 | 55 | Lab 2: Hessian + Newton |
| 2:45 | 30 | Lab 3: damping & failure |
| 3:15 | 20 | Lab 4: regularizer, ridge for free |
| 3:35 | 20 | Debrief |

**Lecture.** Newton from Taylor: `Hp=−g`; quadratic ⇒ one step; local **quadratic** convergence. GLM Hessian `XᵀDX/n`, `D=diag(φ″)`: linear `D=I` (one step = normal equations); logistic `D=diag(σ(1−σ))`; **IRLS** = Newton as reweighted least squares. **Statistics link**: the Hessian is the observed **Fisher information**; its inverse estimates `cov(ŵ)`. **Cholesky** `A=LLᵀ` (`n³/3`) succeeds **iff** SPD; failure ⇔ not positive definite. Damping (line search on the Newton step) and `H+τI`; ridge `H+λI` is always SPD.

**Worked examples.** Newton 1-D `eˣ−2x`: `1→0.7358→0.6940→0.693147` (digits double); Cholesky by hand `[[4,2,2],[2,5,3],[2,3,6]]→L=[[2,0,0],[1,2,0],[1,1,2]]`; Cholesky fails on `[[1,2],[2,1]]` (`L₂₂²=−3`); separable Iris, `λ=0` ⇒ `‖w‖→∞`.

**Lab 1 — Cholesky.** `CholeskySolver` (`solve(A,b)` via `LLᵀ`); `cholesky`, `solve_lower`, `solve_upper_from_lower`; `NotPositiveDefiniteError` provided; vectorize over `i`. Tests: **`ILinearSolver` contract test**; vs `np.linalg` (oracle); negative eigenvalue → error; Hilbert (residual small, error large).

**Lab 2 — Hessian & Newton.** `GLMLoss.hessian=XᵀDX/n`; `check_hessian` via `numerical_jacobian` (DRY). `NewtonDirection(linear_solver)`; `newton()` assembly. **No loop edit**. Tests: linear in **1** iter vs `np.linalg.solve`; logistic same minimizer as GD/sklearn; `e_{k+1}/e_k²` bounded; IRLS check.

**Lab 3 — Damping & failure.** Damped Newton = `NewtonDirection` + `Armijo` (`α=1`). Double well from `x₀=0.3` (indefinite Hessian, saddle); remedy `H+τI`, `τ` doubled. Tests: strict decrease → `x=±1`; exception without damping.

**Lab 4 — IRegularizer, ridge for free.** `L2` (`value ½λ‖w‖²`, `gradient λw`, `prox w/(1+λt)`), `NoRegularizer`, `RegularizedObjective(loss, reg)`. Ridge = any optimizer on `RegularizedObjective(GLMLoss, L2)` — live OCP; tie to MAP (Gaussian prior). Tests: wrapped gradient/Hessian vs numerical; ridge Newton vs closed form.

**SOLID / NB / Plan B / Q.** O: Newton and ridge added with no edits. D: `NewtonDirection` receives its solver. NB: Newton log-log error; double-well field; ridge path. Plan B: ship the `cholesky` loop. Q: why not invert `H`? what does "Cholesky failed" mean geometrically **and** statistically?

---

### Day 5 — Nonlinear least squares: Gauss–Newton and Levenberg–Marquardt
**Headline.** Curve fitting: exploit the least-squares structure. Gauss–Newton approximates the Hessian by `JᵀJ`; Levenberg–Marquardt damps it — and LM *is* a trust-region method. Reuses Cholesky from Day 4.

**Deliverable.** `ILeastSquaresProblem` instances (`curve_fitting.py`); `GaussNewton`; `LevenbergMarquardt` (with the gain-ratio update).

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | Lecture (incl. the trust-region naming) |
| 0:55 | 45 | Lab 1: `ILeastSquaresProblem` + Jacobian check |
| 1:40 | 45 | Lab 2: Gauss–Newton |
| 2:25 | 70 | Lab 3: Levenberg–Marquardt |
| 3:35 | 20 | Debrief |

**Lecture.**
- **Nonlinear least squares** `min ½‖r(x)‖²`, `r: ℝⁿ→ℝᵐ`. Gradient `g=Jᵀr`, exact Hessian `∇²f=JᵀJ + Σ rᵢ ∇²rᵢ`.
- **Gauss–Newton** drops the second term (valid for small residuals or near-linear models): solve `(JᵀJ)δ=−Jᵀr`. Cheap (only first derivatives), near-quadratic when residuals are small, but `JᵀJ` can be singular and GN can diverge.
- **Levenberg–Marquardt** solves `(JᵀJ + λI)δ=−Jᵀr`: `λ→0` = Gauss–Newton, `λ→∞` = a small gradient step. `JᵀJ+λI` is **SPD for λ>0**, so Cholesky never fails — a direct payoff of Day 4.
- **Trust region (10 min, naming what they code).** LM's `λ` is the dual of a trust-region radius `Δ`: bigger `λ` ⇔ smaller trusted step. The **gain ratio** `ρ = (actual reduction)/(predicted reduction)`, predicted `= ½δᵀ(λδ − g)`, drives the update: `ρ` small/negative ⇒ reject the step, increase `λ` (shrink the region); `ρ` large ⇒ accept, decrease `λ`. This is exactly the Day-4 damped-Newton `H+τI` idea, now with a principled `τ`. Distinction to give: **line search** fixes the direction and searches the length; **trust region** fixes a radius and searches direction *and* length inside the ball.

**Worked examples.**
- Exponential decay `y=a·e^{−bt}+c`, `(a,b,c)=(2.5,1.3,0.5)`, `t∈[0,4]`, noise `0.05`. From `(1,1,1)`: GN converges in a few steps. From `b=10` (model near-constant past `t≈0.5`): `a,b` poorly determined, `JᵀJ` near-singular ⇒ GN drifts, LM copes.
- Two near-equal exponential rates ⇒ near-singular `JᵀJ` ⇒ GN fails, LM robust.
- **LM is not global**: `y=a·sin(ωt+φ)` with `ω` far off ⇒ converges to a *local* minimum. "Converged" ≠ "best".

**Lab 1 — `ILeastSquaresProblem` + Jacobian check (45 min).**
```python
class ExpDecay:      # ILeastSquaresProblem: residuals(x)=model(t;x)−y, jacobian(x)
class GaussianPeak:  # another instance
```
Write `residuals` and analytic `jacobian`; validate `jacobian` with `numerical_jacobian` (Day 4, DRY). Tests: Jacobian vs numerical; residual/gradient consistency `g=Jᵀr` vs `check_gradient` on `½‖r‖²`.

**Lab 2 — Gauss–Newton (45 min).**
```python
class GaussNewton:   # IOptimizer: minimize(problem, x0); __init__(linear_solver)
```
`δ = solve(JᵀJ, −Jᵀr)` via the **Day-4 `CholeskySolver`** (DIP — reused, not rewritten), then `x += δ` (optionally with Armijo on `½‖r‖²`). Tests: recover exact parameters on noiseless data (`1e-8`); converges from a good start; **diverges** from the hard start (documented failure).

**Lab 3 — Levenberg–Marquardt (70 min).**
```python
class LevenbergMarquardt:   # IOptimizer: __init__(linear_solver, lambda0=1e-3, ...)
```
Damped solve `(JᵀJ + λI)δ = −Jᵀr` (dense `+λI`), gain ratio `ρ`, accept/reject, update `λ`. Because `JᵀJ+λI` is SPD, Cholesky always succeeds. Tests: converges where GN failed (hard start); recovers parameters; matches `scipy.optimize.least_squares(method="lm")` (oracle); on NIST StRD, hits the certified values from the "hard" start.

**SOLID of the day.** ISP: `ILeastSquaresProblem` is its own interface (`residuals`, `jacobian`) — **not** an `IObjective`; GN/LM ask only for what they need. DIP: both receive an `ILinearSolver`; the Day-4 Cholesky is reused unchanged. O: GN and LM are new `IOptimizer`s; nothing existing is edited. S: the problem models the data, the solver factorizes, the optimizer iterates.

**Notebook 05 / Plan B / Q.** NB: GN vs LM trajectories on the fit; evolution of `λ` and `ρ`; residual curves. Plan B: if behind, do LM only (GN is the `λ=0` limit) and skip NIST. Q: why does GN drop the second Hessian term, and when is that dangerous? why is LM naturally compatible with Cholesky while GN is not? how does the gain ratio decide the step?

---

### Day 6 — Regularization, sparsity, and robustness (proximal, lasso, Huber)
**Headline.** Ridge was smooth; **lasso is not** — and non-smoothness is exactly what yields sparsity. Meet the proximal gradient method, then robust regression. Close with a benchmark and a code review.

**Deliverable.** `L1`, `ElasticNet`; `ProximalGradient` (ISTA, optional FISTA); `Huber`, `PoissonNLL`; benchmark; extensibility challenge.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | Lecture |
| 0:55 | 65 | Lab 1: proximal gradient & lasso |
| 2:00 | 40 | Lab 2: Huber (robust) & Poisson |
| 2:40 | 40 | Lab 3: benchmark |
| 3:20 | 35 | Lab 4: extensibility challenge |
| 3:55 | 20 | Debrief: cross code review |

**Lecture.** Why smooth methods fail on `‖w‖₁` (non-differentiable at 0, and that kink pins coordinates to exactly 0 → sparsity). **Proximal operator** `prox_{tr}(v)=argmin ½‖w−v‖²+t r(w)`: L1 → **soft-threshold** `sign(v)·max(|v|−t,0)`; L2 → shrinkage `v/(1+t)`; 0 → identity. **Proximal gradient (ISTA)** `w⁺=prox_{αr}(w−α∇smooth)`, `α≤1/L`; **FISTA** adds momentum (bonus). Sparsity = feature selection = Laplace prior (MAP). **Robust regression**: squared error is outlier-sensitive; **Huber** is quadratic near 0, linear far out (bounded influence) — and *smooth* (contrast with L1's non-smoothness in the parameters).

**Worked examples.** Soft-threshold at `v=(3,−0.4,0.1)`, `t=0.5` → `(2.5,0,0)`; 1-D lasso `½(w−3)²+λ|w|` → `w=soft(3,λ)` (`2.5` at `λ=0.5`, `0` at `λ≥3`); Huber vs squared under one gross outlier.

**Lab 1 — Proximal gradient & lasso (65 min).** `L1` (`prox`=soft-threshold; `gradient` raises), `ElasticNet`; `ProximalGradient(smooth, reg, step)` (ISTA; FISTA optional). **Reuses `GLMLoss` unchanged**; lasso = `ProximalGradient(GLMLoss, L1)` (OCP). Tests: **`IRegularizer` contract test** (prox identities); hand soft-threshold; vs scikit-learn `Lasso` (oracle); recover a known sparse `w*`; `L1.gradient` raises. Experiment: the **regularization path** on *diabetes*.

**Lab 2 — Huber & Poisson (40 min).** `Huber`, `PoissonNLL` as `IPointwiseLoss` ⇒ GD/Newton work on them **for free** (OCP payoff). Tests: derivatives vs numerical; `Huber → least squares` as `δ→∞`; robust fit vs outlier; Poisson vs scikit-learn `PoissonRegressor`.

**Lab 3 — Benchmark (40 min).** `benchmarks/run.py` → CSV → notebook 06. Problems: ill-conditioned quadratic, Rosenbrock, linear (raw California), logistic (breast cancer, digits), curve fit (NIST), lasso (diabetes), a large logistic for stochastic. Methods: GD+Armijo, momentum, SGD, Adam, Newton, ridge, GN, LM, lasso. Count iterations/epochs/evals/wall-clock/final loss vs oracles. **One-pager** "which optimizer for which problem?" (graded).

**Lab 4 — Extensibility challenge (35 min).** Add one new implementation of an existing interface, **editing no existing `src/` file** (checked by `git diff`): `FISTA`, `GroupLasso` (block prox), `Nesterov` (`IDirectionRule`), or `PoissonNLL` if not done. Justify any forced edit in `DESIGN.md`.

**Debrief — cross code review (20 min).** Checklist over a peer's repo: SRP, ISP, LSP (contract tests pass on all implementations?), OCP/DIP, DRY, tests, `mypy --strict`, error messages. Retrospective: which interface made an addition trivial; where the loss-as-likelihood view paid off.

**SOLID / NB / Plan B / Q.** O: lasso, Huber, Poisson added without edits. S: penalty fully separated from loss. I: `ProximalGradient` needs only `prox`. NB: soft-threshold shape; `w(λ)`; lasso path; Huber vs squared; benchmark profiles. Plan B: skip FISTA/ElasticNet, keep ISTA+L1. Q: why does L1 create sparsity? what is a proximal operator? why is Huber robust yet smooth?

---

## 6. Recurring problem gallery
| Problem | Days | Shows |
|---|---|---|
| Quadratic `½xᵀAx−bᵀx`, tunable `κ` | W2 1,2,4 | conditioning, zigzag, rate, Newton in one step |
| Rosenbrock | W1 project, W2 2,4 | gradient check, curved valley |
| `eˣ−2x`, double well | W2 4 | Newton digits; indefinite Hessian, Cholesky failure |
| Toy linear/logistic, separable Iris | W2 1,4 | hand tests; divergence without regularization |
| California, breast cancer, digits | W2 1,2,4 | conditioning, GD/Newton, `w`-as-image |
| Ill-scaled data (1 & 1000) | W2 1,3 | conditioning; Adam's per-coordinate scaling |
| a9a / covertype | W2 3,6 | realistic-scale stochastic |
| Exp decay / Gaussian peak / sinusoid / NIST StRD | W2 5,6 | GN/LM, singular `JᵀJ`, local minima, certified values |
| diabetes, synthetic sparse | W2 6 | ridge vs lasso, regularization path |
| Outlier-contaminated regression | W2 6 | Huber vs least squares |
| Babylonian root, `x·eˣ`, XOR net | W1 project | autodiff (through a loop, on a model) |

---

## 7. Assessment, risks, and the still-cut material

**Assessment.** Hidden tests 40 % · code quality (`mypy`, `ruff`, no dead code, no scipy in `src/`) 20 % · extensibility challenge 15 % · `DESIGN.md` 10 % · benchmark note 15 %.

**Risks & fallbacks.** Too much material → cut in order: FISTA/ElasticNet, Poisson, heavy-ball theory, then GN (keep LM as the `λ=0`-inclusive method); never the loss/MLE thread. Students behind → interfaces provided, pairs, one reference module on request. Fragile numeric tests → fixed seeds, calibrated tolerances. No network → environment + data in the clone. Day-3 dataset too big → ship a subsample.

**Still cut: CG / PCG / sparse matrices.** The freed Week-2 day went to GN/LM, so CG/PCG did not fit — a real loss for large sparse systems, acknowledged. If a future edition finds room (a 7th day, or trimming the "plus" items), the clean insertion order is: sparse CSR → CG → Jacobi-preconditioned CG → Newton-CG. It slots in without editing existing code, because `NewtonDirection`, `GaussNewton`, and `LevenbergMarquardt` already depend on the abstract `ILinearSolver`: a CG solver is simply another implementation, and Newton-CG is Newton with it injected.

---

## 8. References
- J. Nocedal, S. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006 (Newton, GN, LM, trust region, quasi-Newton).
- S. Boyd, L. Vandenberghe, *Convex Optimization*, Cambridge, 2004.
- A. Beck, *First-Order Methods in Optimization*, SIAM, 2017 (proximal, ISTA/FISTA).
- T. Hastie, R. Tibshirani, M. Wainwright, *Statistical Learning with Sparsity*, CRC, 2015 (lasso, paths).
- L. Bottou, F. Curtis, J. Nocedal, "Optimization Methods for Large-Scale Machine Learning", *SIAM Review* 60(2), 2018 (SGD, Adam).
- A. Griewank, A. Walther, *Evaluating Derivatives*, SIAM, 2008 (forward/reverse autodiff — Week 1).
- W. Squire, G. Trapp, "Using Complex Variables to Estimate Derivatives of Real Functions", *SIAM Review* 40(1), 1998 (complex step — Week 1 bonus).
- NIST StRD, *Statistical Reference Datasets — Nonlinear Regression* (certified GN/LM benchmarks).
- A. Karpathy, *micrograd*, 2020 (reverse-mode engine — Week 1).
