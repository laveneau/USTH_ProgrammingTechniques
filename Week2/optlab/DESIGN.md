# DESIGN.md

Your design log. Fill it in as you go - it is graded (10 %).

## 1. Decisions

For each non-obvious choice you make, one short entry: what you chose, what you
rejected, and why. Two or three sentences each.

> Example (day 2) - *`History` stores a list of `StepEvent` rather than parallel lists
> of values and gradient norms. A `StepEvent` is already the unit the `IObserver`
> interface hands over, so unpacking it into parallel arrays would duplicate its shape
> in a second place and break whenever a field is added.*

## 2. SOLID, where it actually paid off

One entry per principle, naming the concrete moment in *your* code:

- **S** -
- **O** -
- **L** -
- **I** -
- **D** -

## 3. Extending optlab (day 6, Exercise 4)

The two classes you added, and what the notebook's comparison cell listed: the new files,
and the existing files you had to edit. Then, for each class, one short paragraph per
principle:

- **O** - which existing files did you edit, and why could you avoid it, or not?
- **L** - does it pass every contract test of its interface? If not, which promise can it
  not keep, and does any optimizer rely on that promise?
- **I** - did the interface make you write a method that can only raise?
- **D** - which existing class receives it, and does that class need to know which one it got?

Where the design failed, say exactly which interface was too narrow and what you would
change in it.

## 4. Anything you would do differently

Short retrospective. Which interface made an addition trivial? Where did the
loss-as-likelihood view change what you wrote?
