# DESIGN.md

Your design log. Fill it in as you go — it is graded (10 %).

## 1. Decisions

For each non-obvious choice you make, one short entry: what you chose, what you
rejected, and why. Two or three sentences each.

> Example (day 2) — *`History` stores a list of `StepEvent` rather than parallel lists
> of values and gradient norms. A `StepEvent` is already the unit the `IObserver`
> interface hands over, so unpacking it into parallel arrays would duplicate its shape
> in a second place and break whenever a field is added.*

## 2. SOLID, where it actually paid off

One entry per principle, naming the concrete moment in *your* code:

- **S** —
- **O** —
- **L** —
- **I** —
- **D** —

## 3. The extensibility challenge (day 6)

Which implementation you added, and the output of `git diff --stat src/` proving no
existing file under `src/` was edited. If an edit *was* forced, say exactly which
interface was too narrow and what you would change about it.

## 4. Anything you would do differently

Short retrospective. Which interface made an addition trivial? Where did the
loss-as-likelihood view change what you wrote?
