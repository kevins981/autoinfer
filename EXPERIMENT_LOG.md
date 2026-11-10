# Experiment log (AutoInfer design experiments)

What we tried on the system itself (modules, team types, models), what happened, and what we
think it means. Newest last. Details live in each run's journal.md and in design/.

## 2026-11-10 · test-6 · moonshot teams (flavor v2, after commit ffcc7b3)
Setup: moonshot = history-free leader + venture funding + sparring + inspiration cards; history
pointers (no results) only after a proposal. Champion T31-003, 1088.9 tok/s.

| Team | Leader | Extra modules | Upside proposed | Outcome | GPU-min |
|---|---|---|---|---|---|
| T35 | Opus 5.5 | none | +35% (run-ahead weight streamer, megakernel) | killed at stage 1 by its own criterion: streamer slows every GEMM | 20 |
| T36 | Fable 5.2 | none | +8-9% (confidence/load-based draft depth) | withdrew after history pointers; closed | 23 |
| T37 | Fable 5.2 | speed of light + ownership line | +18% (fixed-size step: deeper drafts + prompt tokens as rows) | stage 1 (depth unlock) +1.34% same-job ABBA, official 1142.2; found a bug in earlier teams' two-slot state code; stages 2-3 not yet run | ~37 so far |

Observations:
- **T37 (Fable + speed of light + the other modules) did best so far:** the most ambitious multi-stage
  plan, and the first win since T31. One team only, and the speed-of-light page isn't cited in its
  proposal, so the credit can't be split between the page, the model and chance.
- **All three leaders converged on the same physics** (memory-bound decode, ~80% fixed cost per step,
  slow PCIe all-reduce) and mostly on ideas earlier teams had tried.
- **History pointers deflate ambition:** T36 withdrew and T37 cut its upside from +18% to +6% right
  after reading them. Open question: reveal history later (after the first milestone) or on request.
- **Sparring helped every time** (new measurements: T35's residual accounting, T36's all-reduce phase
  breakdown) but didn't reverse a closing decision. T37 ended its conversation after one question.
- **Leaks in "history-free":** `agent list` (other teams' tasks), record ids (T31 implies ~30 teams),
  and rulings in task_definition.md (who asked for what).
- **Official vs same-job:** T37-002's official +4.9% vs ABBA +1.34%: the champion's official number is
  from an older job (before a reboot); trust the same-job comparison.
