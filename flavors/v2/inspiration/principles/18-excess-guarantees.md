# Excess guarantees

**The principle.** Every part of a system makes promises to whatever uses it: about order, exactness, freshness, completeness, timing. Each promise costs work to keep. Promises are usually set strong, because strong is safe and the users' real needs weren't known. Often what the users actually rely on is weaker.

The promises a system makes to the outside world are set by its requirements. The promises its parts make to each other are choices.

**The waste.** Work spent keeping a promise nobody relies on.

**General directions.**
- **Ask who relies on it:** for each promise, find what would break without it.
- **Weaken what nobody relies on:** while keeping everything the outside depends on.
- **Keep it only where it matters:** at the points or moments where it is relied on.
- **Keep it more cheaply:** the same promise, delivered by a different mechanism.

**Illustrations** (examples only; the best use of this principle may look like none of them). Not every letter needs registered mail. A bank need not post each payment the instant it is made. A list need not be fully sorted to find its largest entry.
