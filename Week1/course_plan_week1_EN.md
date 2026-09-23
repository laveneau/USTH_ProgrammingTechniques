# Programming Techniques — Week 1 of a 50 h course

**MSc Data Science · 25 h · 6 days × ≈ 4 h 15 · foundations for Week 2 (Optimization)**

> Language of instruction: **English**. Code identifiers are English.

---

## 0. Where this week sits

This is **Week 1: clean, modern Python** — how to write code that is correct, typed, tested, and well-architected. It is the prerequisite for **Week 2: Optimization**, where students extend a small library (`optlab`) under exactly these disciplines.

The week is **four lecture + labwork units** followed by a **project**, across six sessions:

| Unit | Theme | Lecture | Labwork | New skill |
|---|---|---|---|---|
| 1 | **Classes** | L1 | Labwork 1 | objects, inheritance, encapsulation, polymorphism |
| 2 | **Typing** | L2a + L2b | Labwork 2 | type hints, duck typing, `mypy` |
| 3 | **Tests** | L3 | Labwork 3 | `unittest`, mocking, `pytest` |
| 4 | **Architecture** | L4 | Labwork 4 | SOLID, TDD, DRY, KISS |
| P | **Project: Automatic differentiation** | project | project | operator overloading, `@dataclass`, the bridge to Week 2 |

> **Pacing — five items over six sessions.** There is deliberately no fixed activity for the spare session, because in practice it is already spoken for: a unit routinely runs a little past its session (Typing and Architecture most often), and the autodiff project comfortably fills more than one. Expect the four units to take **four to five sessions** and the project the remaining **one to two**. Let that boundary float with the real pace rather than cutting a labwork short to stay on a schedule — the per-unit timings below describe the rhythm of one session, not a timetable. The project **replaces the former end-of-week project** and is the hinge into Week 2.

### Goals of the week
By the end of the week a student can: design a small class hierarchy; annotate it and pass `mypy --strict`; write `unittest` tests including mocks, and read and write the `pytest` equivalents; refactor code to satisfy each SOLID principle; and apply all of this in a non-trivial project (a working automatic-differentiation engine).

### Tooling (installed in the first session, used all week)
Python ≥ 3.11, `mypy` (run `--strict` from Unit 2), `unittest` (standard library -- what the labworks and the project actually run) and `pytest` (installed in Unit 3, used from Week 2 on), `ruff` (optional), `colorama` (Labwork 3). A single project layout is introduced with the project and reused in Week 2.

### Session rhythm (≈ 255 min)
Lecture 60–90 min · break 10 · guided lab (write / refactor / test code) ≈ 2 h 30 · debrief 20 min. Labs are done **at the keyboard**, each exercise finishing on code that runs (and, from Unit 3, on green tests).

Unit 2 is the exception: its lecture is delivered in **two halves of 40 min** with lab time between them (see below). No unit puts more than 90 minutes of lecture in front of a keyboard.

### Materials review — corrections applied
Checking the syllabus against the actual notebooks surfaced fixable issues; corrected files accompany this plan.

- **Labworks (cleaned starters)** — uniform `%%writefile` + `!python` (+ `!mypy --strict` from L2), consistent `labworkN/` layout, incidental bugs fixed, *pedagogical* flaws kept (the `Dog` shared-list bug; LW3's planted errors; LW4's "before" code). LW2's `power` drops the needless numpy.
- **Corrections (fixed)** — real defects repaired: LW1 the mutable default `tricks=[]` and `miles_to_km` returning a **tuple** (`1,852` → `1.852`); LW2 numpy removed and a redundant `Union` type simplified; LW4 an unused import, an untyped/non-abstract `ICoffeeShop.name`, a parameter named `str` shadowing the builtin, and an `Optional` dereferenced without a guard (the last two broke `mypy --strict`).
- **LW3** ships a dedicated **correction** whose tests reveal the planted errors, sorted into *behavioural* (caught by tests), *specification* (docstring vs code), and *static* (caught by `mypy --strict`) — a good lesson that tests and the type checker are complementary.
- **Closed**: the YAGNI gap is fixed — Lecture 4 now carries a `## YAGNI` section, so it agrees with Labwork 4's `# YAGNI` payoff line.
- **Lecture 2 split** — the one lecture that was out of proportion (~6 800 words, 3 code cells, ~45 min of solid reading) is now **`Lecture2a.ipynb`** and **`Lecture2b.ipynb`**, taught either side of the first half of Labwork 2. All 28 original content cells were carried over unchanged; what was added is each half's own framing plus 9 runnable cells, two of which invoke `mypy --strict` so students see it object to a program that runs. The original `Lecture2.ipynb` is still on disk — **delete it once you have reviewed the split**.

---

## Unit 1 — Classes in Python

**Goal.** Model a domain with classes: state, behavior, inheritance, and Python's object model.

**Deliverable.** Labwork 1 completed: the `Dog`/mutable-default fix, a `Vegetable` class, a `Vehicle` hierarchy, a multiple-inheritance scheme with `Enum`, and a `DoubleLinkedList` with an iterator.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 75 | Lecture |
| 1:15 | 10 | Break |
| 1:25 | 30 | Setup: Python, editor, `mypy`, repo |
| 1:55 | 100 | Labwork 1 |
| 3:35 | 20 | Debrief |

#### Lecture (L1)
Class vs object; `self`; the constructor `__init__` (default and parameterized); **class variables vs instance variables** (the classic mutable-default trap); destructors. **Inheritance**: parent/child, `object`, calling `super().__init__()`, multilevel and multiple inheritance. **Encapsulation**: protected `_x` and "private" `__x` (name mangling). **Polymorphism**: built-in and user-defined polymorphic functions, polymorphism with methods and with inheritance. **Static and class methods** (`@staticmethod`, `@classmethod`).

#### Labwork 1 (exercises)
1. **Spot the bug**: `Dog.tricks = []` as a *class* variable is shared across instances — fix it by moving the list into `__init__` (instance state). The canonical class-vs-instance lesson.
2. **`Vegetable`**: a minimal class with a `name`, instantiated and printed.
3. **`Vehicle`**: capacity, engine-powered or not, top speed; then `Bicycle`, `Car`, `Boat`, `Plane`.
4. **Multiple inheritance**: a `Vehicle` class and an `Engine` class (fuel type as an `Enum`, average consumption) combined by multiple inheritance — `Car`/`Boat`/`Airplane` are `class X(Vehicle, Engine)`; `Donkey`/`Bicycle` have no engine.
5. **`DoubleLinkedList`**: empty constructor, `__iter__`, prepend, remove-first; a test program that builds, prints forward and reverse, removes, and checks.

#### Notes / pitfalls
Mutable class attributes; forgetting `super().__init__()`; the difference between `_x`, `__x`, and public; when a `@staticmethod` should really be a free function.

---

## Unit 2 — Typing

**Goal.** Add static type information to dynamic Python, understand what the type checker can and cannot prove, and make `mypy` part of the workflow.

**Deliverable.** Labwork 2 completed: fully annotated functions and classes (including the `Vegetable` and the multiple-inheritance scheme from Unit 1) and an annotated linked list, all passing `mypy --strict`.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 40 | **Lecture 2a** — annotating the code you write |
| 0:40 | 45 | Labwork 2, exercises **1–2** |
| 1:25 | 10 | Break |
| 1:35 | 40 | **Lecture 2b** — generics, classes, the `typing` module |
| 2:15 | 80 | Labwork 2, exercises **3–4** |
| 3:35 | 20 | Debrief + `mypy --strict` from here on |

#### Lecture, in two halves (L2a, L2b)

Lecture 2 was one notebook of ~6 800 words against 3 code cells — by a wide margin the longest uninterrupted read of the week (L1: 3 300 words, but 29 code cells; L3: 3 700; L4: 4 400). It is now split at its natural seam — *how do I annotate?* / *how does the checker reason?* — which is also where Labwork 2 changes gear. No original material was dropped; both halves gained runnable cells.

**L2a — annotating the code you write** (≈ 20 min read, 7 code cells). **Type systems**: dynamic vs static vs **duck typing**, and why hints have no runtime effect. Type hints for variables, parameters, returns; **sequences and mappings**; type aliases; functions without return (`None`, `NoReturn`). `Any`, and why it is close to no annotation at all. Students **run `mypy` themselves** on a file the notebook writes: the program runs fine and the checker still objects, which is the case for static checking in one screen. A boxed note maps the `List`/`Dict`/`Optional` spellings used throughout onto the modern `list`/`dict`/`X | None` form, since this course requires Python ≥ 3.11.

**L2b — generics, classes, and the `typing` module** (≈ 25 min read + a reference section to skim, 5 code cells). **Type theory**: subtypes; covariant / contravariant / invariant; gradual typing and consistent types. **`TypeVar`** in its three flavours (plain, constrained, `bound=`); `Optional`; type hints for methods; classes as types and forward references; returning `self`/`cls`; annotating `*args`/`**kwargs`; **`Callable`**. A tour of the `typing` module — marked explicitly as a catalogue to skim, not a read. Closing: PEP 8 style.

> The variance section is backed by a second `mypy` run students execute. `Sequence[float]` accepts a `list[int]`, a `list[bool]` and a `tuple[float, ...]`; `list[float]` rejects all three — **but accepts a bare literal `[1, 2]`**, because `mypy` infers a literal's element type from the parameter it is passed to. Hence the lesson, and it generalises well beyond typing: *"my test call passed" is not "my annotation is right"*.

> Bridge note for Week 2: duck typing here is the informal version of the **ABC interfaces** used in Week 2; `Protocol` is its structural-typing form (mentioned, not required).

#### Labwork 2 (exercises)
Exercises 1–2 need only **L2a**; exercises 3–4 need **L2b** — exercise 3 is unreachable without `TypeVar`.

1. Add parameter and return typing to a small `power` function. *(In the original it used `numpy.floor` — numpy's only incidental appearance in Week 1; the cleaned version uses `//` instead. numpy proper is ramped up in Week 2.)*
2. Re-annotate the Unit-1 `Vegetable` class.
3. Play with a linked-list implementation (from realpython) and type it correctly — this is where `TypeVar`/`Generic`/`Optional` earn their keep.
4. Re-annotate the Unit-1 multiple-inheritance exercise (the vehicles).

#### Notes / pitfalls
`Optional[T]` vs `T | None`; invariance of `list`; when a `TypeVar` is needed; `Any` as an escape hatch to avoid; running `mypy --strict` and reading its messages.

---

## Unit 3 — Tests

**Goal.** Make correctness checkable and repeatable: write unit tests, structure them, and isolate behavior with mocks.

**Deliverable.** Labwork 3 completed: tests for every function of a given file, tests for a small class, and a tested class hierarchy where dependencies are **mocked** (a payroll example).

| Time | Len | Activity |
|---|---|---|
| 0:00 | 60 | Lecture |
| 1:00 | 10 | Break |
| 1:10 | 15 | `pip install colorama pytest`, test runner setup |
| 1:25 | 110 | Labwork 3 |
| 3:15 | 20 | Debrief |

#### Lecture (L3)
Why test: automated vs manual, **unit vs integration**. `unittest`: where tests live, structuring a `TestCase`, assertions, testing **side effects**. Running tests and reading output. Advanced: expected failures, isolating behavior, and **`unittest.mock`**. Closing section: the same tests in **`pytest`** -- plain functions and bare `assert`, `pytest.raises`, `pytest.approx`, `@pytest.fixture` / `conftest.py`, `parametrize` and markers -- and the fact that `pytest` runs `unittest.TestCase` unchanged, which is why Week 2 can switch runner without invalidating anything.

#### Labwork 3 (exercises)
1. Add unit tests to a given file, covering **all** functions and all cases (edge cases included).
2. Test a simple class (reused from an earlier labwork).
3. A small class hierarchy plus a function that consumes a list of heterogeneous objects: test each class; then a **payroll** system where the `Employee` classes are **mocked** rather than instantiated.

#### Notes / pitfalls
One behavior per test; arrange–act–assert; testing exceptions (`assertRaises`); when to mock (external or expensive collaborators) vs use the real object; not asserting on implementation details.

---

## Unit 4 — Software architecture: SOLID, TDD, DRY, KISS, YAGNI

**Goal.** Turn "code that works" into "code that stays healthy": each SOLID principle, plus the practices that keep a codebase changeable.

**Deliverable.** Labwork 4 completed: five refactorings, one per SOLID principle, each passing `mypy --strict` and running.

| Time | Len | Activity |
|---|---|---|
| 0:00 | 80 | Lecture |
| 1:20 | 10 | Break |
| 1:30 | 105 | Labwork 4 |
| 3:15 | 20 | Debrief |

#### Lecture (L4)
**S** — Single Responsibility (before/after). **O** — Open/Closed. **L** — Liskov Substitution. **I** — Interface Segregation. **D** — Dependency Inversion. Then **TDD** (red → green → refactor), **DRY**, **KISS**, and **YAGNI** (don't add an abstraction until a concrete need exists — it pairs with KISS and OCP). Interfaces are expressed with `ABC` + `@abstractmethod`, and the lecture states the course's two interface conventions once, where `ABC` is introduced: an interface is named with a leading capital `I` (`IShape`, `INotification`, and all of Week 2's `optlab`), and an interface is **pure** -- every method abstract, no implementation, no state. A class carrying implementation is an abstract base class, keeps its plain name, and is not an interface.

> Materials note: Lecture 4 covers all of SOLID/TDD/DRY/KISS **and** YAGNI, so it matches Labwork 4 (exercise 4), which ends on a `# YAGNI` payoff line.

#### Labwork 4 (exercises) — a coffee-shop running example
1. **SRP**: split a class that mixes data and address-changing responsibilities.
2. **OCP**: replace an `isinstance`-ladder invoice service with polymorphism, so a new company type needs no edit to existing code.
3. **LSP**: fix a subclass (`B.takeaway` raising instead of honoring the contract) so any `CoffeeShop` is substitutable.
4. **ISP**: break a fat `ICoffeeShop` ABC (traditional + third-wave brewing) into segregated interfaces so no class is forced to `raise NotImplementedError`.
5. **DIP**: make `Delivery` depend on abstractions (`ICoffeeShop`, `ICustomer`, `IDelivery`) rather than the concrete `Customer`/`CoffeeShop`; the solution also exposes fluent `set_customer`/`set_coffee_shop` setters.

#### Notes / pitfalls
SRP is about *reasons to change*, not line count; OCP means **add**, don't edit; an LSP violation often hides in a strengthened precondition or a thrown "not supported"; ISP failures show up as `NotImplementedError`; DIP is "inject the collaborator, depend on its interface". These five are exactly the muscles Week 2 exercises every day.

---

## Project — Automatic differentiation (the bridge to Week 2)

**Headline.** Build a small engine that computes **exact** derivatives of Python code, in both modes; reverse mode *is* backpropagation. This project **integrates the whole week** — classes, operator overloading, typing, tests, SOLID — and produces the tool Week 2 runs on.

**New Python here (at the point of use).** **Arithmetic operator overloading** — `__add__`, `__radd__`, `__sub__`, `__neg__`, `__mul__`, `__rmul__`, `__truediv__`, `__pow__` (returning `NotImplemented` for unhandled types) — and **`@dataclass`** / `@dataclass(frozen=True)`. Everything else (classes, typing, `unittest`) is this week's material.

**Deliverable.** `autodiff/dual.py` (forward), `autodiff/tensor.py` (reverse), `autodiff_gradient(f)`, all validated by a finite-difference gradient check. This module is carried into the Week-2 repository and reused there as a gradient oracle.

**Budget: one to two sessions**, depending on how far the four units overran. The natural split:

| Session | Time | Len | Activity |
|---|---|---|---|
| 1 | 0:00 | 30 | Project scaffold (below) + math/design primer (below) |
| 1 | 0:30 | 45 | Lecture (+10-min bonus: three algebras) |
| 1 | 1:15 | 10 | Break |
| 1 | 1:25 | 60 | Part 1: dual numbers (forward mode) |
| 1 | 2:25 | 70 | Part 2 begins: `Tensor` scalar ops + `backward` |
| 1 | 3:35 | 20 | Debrief |
| 2 | 0:00 | 20 | Recap of the graph and the adjoint rule |
| 2 | 0:20 | 95 | Part 2 continues: broadcasting, `matmul`, `sum`, `exp`/`log` |
| 2 | 1:55 | 10 | Break |
| 2 | 2:05 | 60 | Part 3: gradients of real expressions |
| 2 | 3:05 | 30 | Consolidation: `make check` green, code review of the engine |
| 2 | 3:35 | 20 | Debrief + teaser of Week 2 |

> **If only one session is left**, compress to: lecture 45 · Part 1 45 · Part 2 95 (ship `_unbroadcast` and the topological sort pre-written) · Part 3 20 · debrief 20, and drop the three-algebras bonus. Part 2 is the part that must not be cut — reverse mode is what Week 2 uses.

#### Project scaffold
The layout used here and throughout Week 2, set up at the start of the project:
```
project/
├── pyproject.toml          # deps + tool config
├── Makefile                # make test · make check (mypy --strict + tests)
├── src/<package>/          # code
└── tests/                  # unit tests
```
This is the first time the four disciplines arrive as one habit rather than four topics: the engine is typed, `mypy --strict` clean, `unittest`-green (`make test`), and SOLID — and it is graded as such.

#### Math/design primer (30 min)
Just enough to start building: the derivative as a local linear map; the **chain rule** as composition of these maps; why numerical (finite-difference) gradients are only approximate; and the design question — "how do we make a program compute its own derivative?" — which the project answers.

#### Lecture
- Four ways to a derivative: by hand (error-prone), **finite differences** (`n+1` evaluations, approximate), **symbolic** (expression blow-up), **autodiff** (exact to machine precision, bounded cost).
- **Forward mode** with dual numbers `a + bε`, `ε² = 0`: `f(a + bε) = f(a) + f'(a)·b·ε`. Product / quotient / `exp` / `log` rules; one sweep = one directional derivative; a full gradient costs `n` sweeps.
- **Reverse mode = backprop**: a computation graph, adjoints `v̄ = ∂L/∂v`, propagation `grad_parent += grad_out × local_jacobian`, **accumulation** at shared nodes, reverse topological order. A scalar gradient costs `O(1)` sweeps regardless of `n` — the reason neural networks are trainable.
- Array rules (for the `Tensor` engine): `C = A @ B ⇒ Ā = C̄Bᵀ, B̄ = AᵀC̄`; `sum` broadcasts the gradient back; **broadcasting**: forward duplicates, backward sums over the duplicated axes.

#### Bonus interlude (10 min): three 2-D algebras
Numbers `a + b·t` with `t² = s ∈ {−1, 0, +1}`: complex, **dual**, hyperbolic. Only `t² = 0` gives an *exact* first derivative (dual); `t² = −1` gives the `O(h²)` **complex step**. The defining form `Q(a+bt) = a² + s·b²` has signature `(1,1,0)` / **`(1,0,1)`** / `(2,0,0)` — to get `i² = −1` you need a minus; `ε² = 0` a zero. A nice tie between operator overloading and a little algebra; coded in `labs/algebra2.py`, outside the package.

#### Part 1 — Dual numbers, forward mode (45 min)
```python
@dataclass(frozen=True)
class Dual:
    val: float
    der: float
    # __add__, __radd__, __sub__, __neg__, __mul__, __rmul__, __truediv__, __pow__
def exp(x: Dual | float) -> Dual | float: ...
def log(x: Dual | float) -> Dual | float: ...
def derivative(f: Callable[[Dual], Dual], x: float) -> float: ...
def gradient_forward(f: Callable[[Vec], Dual], x: Vec) -> Vec: ...   # n sweeps
```
This is the first real use of **operator overloading** and of **`@dataclass`**. Tests: known derivatives; comparison against a finite-difference gradient; differentiating **through a loop** (Babylonian `√2`: `val ≈ 1.414`, `der ≈ 0.354`). Observe: `n` sweeps for a full gradient → motivates reverse mode.

#### Part 2 — `Tensor`, reverse mode (95 min)
```python
class Tensor:
    data: Vec
    grad: Vec
    def __add__, __sub__, __mul__, __truediv__, __neg__, __pow__   # scalar exponent
    def __matmul__(self, other) -> "Tensor": ...                    # 1-D and 2-D
    def sum(self, axis=None) -> "Tensor": ...
    def exp(self) -> "Tensor": ...
    def log(self) -> "Tensor": ...
    def backward(self) -> None: ...                                 # scalar output
```
Steps: scalar ops + `backward` on a small graph; the **shared node** `x*x + x` (accumulation `+=`); broadcasting via `_unbroadcast`; `matmul` and `sum`; `exp`/`log`; a DRY helper once three ops repeat the same shape. Tests: gradient check per op; shared node; the bias case `XW + b`; matmul shapes; `backward` on a non-scalar raises. Pitfalls: forgetting `+=`; size-1 axes; a **recursive** topological sort that overflows on a long graph (make it iterative).

> SOLID in the project: each node owns its own `_backward` closure, so adding an operation is a new method, not an edit to the engine (OCP). `Dual` and `Tensor` are separate modules (SRP). The scope is kept deliberately small (KISS): no `requires_grad`, no operator registry.

#### Part 3 — Gradients of real expressions (20 min)
```python
def autodiff_gradient(f: Callable[[Tensor], Tensor]) -> Callable[[Vec], Vec]: ...
```
Gradients of `½‖Xw − y‖²` (vs the hand formula `Xᵀ(Xw − y)`) and of the logistic negative log-likelihood. The cost ratio `time(grad)/time(value)` is provided (constant for reverse, growing for forward).

#### Teaser of Week 2 (debrief)
"Next week we stop *computing* gradients and start *following* them downhill to fit real models — linear and logistic regression, then modern stochastic training. You have just built the tool that will check every gradient we write."

---

## Assessment (proposal)

| Item | Weight | Content |
|---|---|---|
| Labworks 1–4 | 40 % | correctness, typing (`mypy --strict`), tests, SOLID refactorings |
| Autodiff project | 50 % | forward + reverse modes, gradient checks green, clean design — one artifact carrying all four disciplines |
| Code quality throughout | 10 % | `mypy`, style, no dead code, readable tests |

---

## Continuity with Week 2

The autodiff module, the project scaffold, and the five SOLID muscles carry directly into Week 2 (Optimization):
- **Interfaces** are `ABC`s, exactly as in Labworks 3–4.
- **Multiple inheritance** (Unit 1) reappears as `GLMLoss(IObjective, ITwiceDifferentiable, IBatchObjective)`.
- **`mypy --strict` + a test suite** are the same `make check` -- the runner changes from `unittest` to `pytest`, which Lecture 3 covers, the habit does not.
- **The autodiff engine** becomes the gradient oracle that validates every hand-written gradient in Week 2.

---

## References
- The four lecture notebooks (Classes, Typing, Tests, Architecture) and Labworks 1–4.
- R. C. Martin, *Clean Code* and *Agile Software Development: Principles, Patterns, and Practices* (SOLID).
- Python docs: `typing`, `unittest`, `unittest.mock`, `dataclasses`, `abc`, `enum`; `pytest` documentation (fixtures, parametrize, markers).
- A. Griewank, A. Walther, *Evaluating Derivatives*, SIAM, 2008 (forward/reverse autodiff).
- W. Squire, G. Trapp, "Using Complex Variables to Estimate Derivatives of Real Functions", *SIAM Review* 40(1), 1998 (complex step — bonus).
- A. Karpathy, *micrograd*, 2020 (reverse-mode engine inspiration).
