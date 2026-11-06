# Flavor v1: TODO

Open design items for this flavor's algorithm (`DESIGN.md`). Items shared by all flavors
(infrastructure, tasks) are in `../../../design/open_questions.md`.

## Algorithm, in planned order
- **Leader behavior**: what exactly a leader does with a direction: reviewing past results
  (BANK, SHARED_KB, its local KB), decomposing into experiments, planning, profiling, choosing
  starting points, deciding when to stop.
- **Validation funnel** (TODO): cheap tests before GPU-heavy ones to save GPU time, e.g. napkin
  math (Amdahl/roofline; record the estimate, since disagreement with the measurement is itself an
  observation) → microbenchmark with real shapes → single batched forward pass + profile → short
  unofficial bench → official eval. Caveats: proxies can mislead (only official numbers count),
  and napkin math can kill good ideas if our model of the system is wrong. Guidance or a gate?
- Evaluation and budget follow-ups (see `evaluation.md`, `budget_and_stopping.md`):
  - Move the test cost table to the task folder when the harness is ported.
  - Default budget size per team per round (CONFIG default) for the qwen task.
- Champion lifecycle: when it changes, comparing candidates built on different starting points,
  `integrate()`.
- **Async "new best version" notification**: v1 snapshots the champion at the start of each
  round. Later: how to tell agents about a new best mid-round without interrupting their work.
- k (directions per round) vs number of leaders. (Continue vs new direction is decided by
  `search_policy.md`.)
- `initialize()`: who does the profile, bound analysis and ledger; whether the ledger review is a
  blocking human gate.
- Human checkpoints (a config knob between human-in-the-loop and autonomous).
- Failure paths: failed review or GSM8K (fix/resubmit?), broken builds, stuck workers or leaders.

## Agent topology
- Barrier rounds vs. rolling (reassign each leader slot as soon as that leader finishes).
  Leaning: barrier for v1.
- Who runs `integrate()` (combining wins from several teams)? Leaning: a dedicated worker
  delegated by the manager.
- Brainstorming details: how many brainstormers and how big the random sample is, whether to use
  other models, the exact briefing contents, and tuning the default `search_policy.md`.
