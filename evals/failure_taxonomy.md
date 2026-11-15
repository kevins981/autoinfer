# Failure taxonomy: why agents miss, kill or under-realize good changes

Provisional. Built bottom-up from concrete cases (raw cases with full evidence:
`evals/failure_cases.md`). Grading a run against the answer key (`evals/EVAL.md`) surfaces candidates:
missed wins, wrong causes, partial gains. Add a type only when a case doesn't fit the existing ones;
merge or split types as cases accumulate. A case can have several types.

Started 2026-11-15 from v4_unofficial_eval/sweep-simple-eval (graded in
`evals/runs/v4_unofficial_eval/sweep-simple-eval/eval.md`).

## 1. Measurement methodology
The experiment itself is flawed, so the number can't support the conclusion: noise, wrong setup,
too little data, a proxy that doesn't reflect the engine.
- Signs: the same thing measured twice disagrees a lot; the record says "noisy", "undiagnosed",
  "inconclusive" and still draws a conclusion; a microbenchmark decides a change the engine never ran.
- Case: host-memory all-reduce (sweep-simple-eval T1-003). Timed on one rank without lining the ranks
  up per sample: 52 vs 92 µs for the same kernel. The record noted "rank skew ... undiagnosed" and
  dropped the change on that number. The in-server test (~5 GPU-min) was never run; other runs got +5%.

## 2. Interpretation
The numbers are fine but read wrongly.
- **Wrong reference point:** compared against the wrong baseline or an assumed floor.
  - Case (same T1-003): compared with NCCL forced to Simple standalone (57 µs), though the team had
    just shown the engine runs NCCL's default at ~72 µs; called 57 µs "the floor" though its own
    13 GB/s figure gives ~40 µs.
- **Wrong cost:** priced the visible kernel time, not the cost the component actually causes.
  - Case (T1 knob table): attention priced as "1% of a step"; its real cost is FlashInfer's per-step
    host planning that idles the GPU ~1-1.4 ms/step (+3.8-6.8% elsewhere). The team later found the
    drafter's half (+2.3%) but not the target's.
- **Noise read as signal:** a gain inside the repeat spread treated as a win (or a loss as a loss).
  - Case (T1-002): NCCL_PROTO=Simple promoted at +1.0% unofficial (spread ~1.6 points); reverted
    later. The first revert reason (kernel names) was also wrong; the NCCL log settled it.
- **Irrelevant comparison:** a loss counted that the design doesn't care about.
  - Case (T1-003): "loses at ≥ 96 rows", though the design routes only small all-reduces to it.

## 3. Path dependence (not revisiting earlier choices)
The diagnosis is right, but the fix stays local: the agent works around a constraint created by an
earlier choice instead of re-opening that choice.
- Signs: "doesn't fit / OOM / not enough X" followed by shrinking the new change rather than asking
  what already holds X.
- Case (T2-002..T2-004): the FP8 draft head (+3% elsewhere) lost because only 15 of 16 requests fit.
  T2's own earlier 2048-token prefill graphs held that memory (another run used 512 and kept the
  head). T2 tried smaller heads instead and never revisited the graph size.

## 4. Never considered
The direction never showed up in the run's maps, ideas or records.
- Signs: absent from the sweep's knob tables and rankings (grep the run's reports for the key node).
- Cases (sweep-simple-eval, stopped at 158 GPU-min, so partly budget): copy-engine all-reduce for
  prefill steps, GDN lazy commit, W8A8 quant fusion, L2 prefetch during the all-reduce.

## Also seen (process costs, not failures of judgment)
- Duplicate work across teams: T1-007 = T2-006 (cancelled after 0.6 min); both teams tested 3 draft
  tokens (5 jobs).

## Counter-examples (good process, for contrast)
- T2-003/T2-007 profiled before spending more starts and found the real cause (lost concurrency, not
  step cost). T1 corrected two of its own explanations when new evidence came.

## Ideas this suggests (to discuss, not adopted)
- A fresh-context verification worker a leader calls before a conclusion or claim: "what is this
  compared against, and is it what the engine runs?", "is the measurement repeatable?", "is this the
  real cost?". Targets types 1 and 2.
- Type 3 needs a different question: "which earlier choice is now the constraint?".
