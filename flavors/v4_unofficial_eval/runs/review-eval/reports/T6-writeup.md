# T6 write-up (leader-6, sweep team; closing)

**Wins: none.** Records: T6-001 (HMAR to 768 rows: microbench 1.7x vs NCCL at 192-512 rows; no server gain shown in one quick start),
T6-002 (row-split overlap, INVALID: double all-reduce), T6-003 (split fixed: +0.28% on T3-007, one start below), T6-004 (split on T5-001:
+0.34% over 3 starts, third start below). GPU: ~39 of 45 GPU-min (profile 2.5, microbench and A/B ~10, numchk ~1, harness 3.2 + 9.1 + 1.0 + 7.5 + 4.4).
Map: reports/T6-sweep.md, T6-sweep/1-6, update-1, update-2. Reviews: T6-1 (map), T6-2 (code, 2 turns), T6-3 (closing decision).

## What I learned
1. **Price collectives without the waiting.** My calibration put the mixed-step AR at 4x its link napkin and ranked it first. T1 attributed most of
   that to rank skew (a hypothesis I did not test directly). Neither a faster transport (HMAR768, one quick start) nor partial overlap (split) showed
   a consistent end-to-end gain. Phase 2 (T1's notes) caught
   this before I spent much, but the framework's cost model has no skew term: it should.
2. **Isolated microbenchmarks can contradict earlier runs.** HMAR beat NCCL 1.7x at 2-5 MB in job 101, contrary to T1's claim. No end-to-end gain was shown (one quick start; the harness screen was never completed).
3. **"Skip the downstream reduction" flags must cover every consumer,** including those outside the layer (final norm, drafter, aux
   capture). MTP acceptance (2.29 vs 2.34) caught a double all-reduce that GSM8K-50 had passed. Acceptance is a cheap correctness signal
   for anything that touches mixed steps.
4. **Re-measuring an exact change on the new champion was the right use of the last budget.** T6-004 missed on its third start by
   0.7 tok/s. Under the rule a gain under 1% needs all 3 starts above the champion; one start below can't be rescued by more starts.

## For the next team
- The split (bank/T6-004, workspace t6-stack) is exact for the tested serialized configuration (bitwise on 6 linear-layer cases; not safe under concurrent ubatching) and stacked cleanly on T5-001; worth re-measuring if mixed steps get cheaper elsewhere.
- HMAR768 (bank/T6-001, workspace t6-arbig) is unmeasured by the harness.
