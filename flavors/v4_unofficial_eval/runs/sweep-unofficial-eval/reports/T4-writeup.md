# T4 write-up (leader-4, worker t4-w1)

Map: [T4-sweep.md](T4-sweep.md), files 1-6 + update-1. GPU: 80 granted, ~13 used (job 199 profile 2.4 min;
microbenches 203/204 0.7 min; engine checks 205-212 ~10 min). No wins, no promotions.

## What we did
- Phase 1 map from the framework and one nsys profile of HEAD (job 199): communication 25% of GPU time (NCCL in mixed
  steps 12.5%, host AR in decode 12.5%), decode GEMMs at 0.79 TB/s, FP8 W8A8 at ~180+ TF in big mixed steps (so the
  4090's FP8/fp32-acc peak is 330, not SPEED_OF_LIGHT's 165).
- My phase-1 top pick (duplex host AR) had already been measured negative by T1/T3: stopped before spending GPU.
- Asked the never-asked ruling on fp16 accumulation in W8A8 (T1/T2 had parked it): **not allowed** (human, final).
- New candidate (P4): layer-level two-half overlap of mixed-step all-reduces. Gated by two cheap microbenches:
  NCCL halves overlap nothing (job 203); host-AR halves beat HEAD's serial NCCL by 1.4-2.9 ms per 8-layer stack,
  bitwise equal (job 204) => ~11-23 ms per mixed step (~4% napkin).
- Built it in the engine (bank/T4-001): the mixer core stays on the full batch; only the row-independent chain
  between mixers is forked. Works eager and in graph capture; 128K passes. **Correctness unresolved**: split-vs-plain
  hidden states differ (max ~100, mean ~0.3) at 531/1159 rows but are bitwise equal at 2048; not tile configs.
  No A/B was run before the run ended.

## What I learned
- Read the bank before the first microbench: my P3 duplicated T1/T3 (caught before GPU spend, but only by luck of timing).
- "Overlap doesn't work on this box" was a statement about NCCL. The SM-driven host AR overlaps. Prior negatives
  measured with one comm path don't transfer to another.
- A microbench with a stand-in for the serial dependency (mixer event) is a cheap, decisive gate (0.7 GPU-min for both).
- Put the exactness check that distinguishes "legit reordering" from "race" into the design up front: here, forcing
  identical kernels per half (incl. cuBLAS bf16 paths) so split-vs-plain is expected bitwise equal.
- Ask rulings early: the fp16-acc question sat parked by two teams.

## Open (for whoever continues)
1. T4-001: strip debug hooks, localize the deviation per layer, then A/B + unofficial_eval. Stacks with T5's C1.
2. Decode W8A16 kernel redesign (~3-4%), lm_head GEMV (~0.8%).
