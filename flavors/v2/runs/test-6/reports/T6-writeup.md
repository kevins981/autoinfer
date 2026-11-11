# T6 write-up (leader-6, explore). Recommending a voluntary close at resync 1

GPU: 28.8 of 60 (20 seed + 40 grant) used; 31.2 to return. Records: bank/T6-001..T6-007 (T6-006 INVALID).
Worker: leader-6.w1. Champion reference throughout: T1-042, 942.2 tok/s.

## Direction: rows above 48 per step (proposals/T6-1.md)
- Premise: rows above M=48 are expensive because of the GEMM kernel, not the hardware.
- Plan:
  1. a W8A16 GEMM that streams the weights once for M=49-256;
  2. use the decode slack it frees, either for T5's deeper speculation or for chunked-prefill piggybacking.
- Forecast: payoff if it works +5-6%, central +1.5%.
- Outcome: the kernel works in microbench. End to end it is ~0 at k=2, and both consumers of the slack are dead.

## What was measured

### Marlin vs roof (T6-001, job 31; cold, per-rank shapes, per-step sum)
- Memory roof 12.74 ms (955 GB/s); compute 0.183 ms/row (133 TF on GPU0).
- Marlin per step: M=48 15.5, M=52-64 17.65, M=72 25.4, M=96 28.1, M=112 29.7, M=128 25.9 ms.
- Cause: marlin_mm splits M into 64-row launches, and each launch streams all 12 GB.
- This is not KB44/T1-032. That record merged the remainder launch but kept serial slabs, and is not in the champion.

### Kernel fixes, all exact to <= 1 bf16 ulp (fp32 reduction order), bitwise at M<=48, guard-tested at M=1..256
- **Parallel slabs in one launch (T6-002):** -3..-8.6 ms at M=72-120 cold.
  - gate_up and down still read the weights twice: the slabs are not time-aligned on the same stripe, and these
    matrices are larger than the 72 MB L2.
- **Slab-inner tile order (T6-003):** DRAM traffic drops to exactly one weight stream (ncu: 134.8 -> 92.0 MB at M=96).
- **Per-shape policy (T6-004, ws t6-pol, AI_POLICY):**
  - Cold: M=72-120 -21..-35%; M=96 18.0 ms = 1.02x roof.
  - Warm: -13..-25%.
  - M=49-64: only -3%. M<=48: unchanged.
- **Occupancy probes (job 46):** 2 CTAs/SM and 8-warp tiles: no gain at M<=48.
  - Stage-3 kernels with thread_k=64 are **wrong**: the group-scale prefetch is gated on pipe%2. Don't use them.

### In-server
- **Policy alone at k=2 (T6-005, job 54):** within noise (888.5/888.3 vs 895.0, n=1).
  - At the 2048-token budget, few steps land in M=49-190.
- **Piggyback with chunk budget 96 (T6-005):** -30% (TPOT 16->21 ms, TTFT x3.3).
- **nsys of a piggybacked mixed M=96 step (T6-007, job 57, built):**
  - Kernel time ~42 ms vs 24.6 for a decode step. In-situ Marlin is +9.1 ms for the 48 extra rows, far above the
    cold microbench's +2.4. Plus hostar +3.3 and GDN prefill glue +3.5.
  - So a piggybacked prefill row costs ~0.36 ms, more than the ~0.25 ms in whole-prefill mixed steps, even with zero
    launch gaps.
  - Mixed steps are also partly launch-bound: 2583 vs 1041 kernels per step; idle 34 ms under nsys, inflated.
- **T5 in situ (T5-006, KB63):** the policy recovers only 0.30 ms of the +3.4 ms M=64 cliff. T5 closed.

## Forecast vs outcome
| part | forecast | measured |
|---|---|---|
| kernel alone, mixed steps (k=2) | +1.0-1.5% | ~0 (T6-005, within noise) |
| slack used by T5's k=3 | +3% (shared) | dead: M=64 cliff stays +3.4 ms in situ (T5-006) |
| slack used by piggyback | +2-5% | -30% at budget 96; kernel floor ~0.36 ms/row (T6-007) |

Main error: I trusted the cold microbench and the CPU sim for the slack.
- In situ, the M>48 rows cost 2-4x more than cold: L2 prefetch, GPU0 contention, hostar bytes.
- The mixed step's non-GEMM fixed costs (GDN prefill glue, eager launches) are much larger than the 1-3 ms the sim
  needed.
- A 15-minute anatomy of a piggybacked step (T6-007) should have come before the kernel work.

## Reusable
- **ws t6-pol / T6-004:** exact Marlin policy for M=65-191, a single-launch masked parallel-slab mode and slab-inner
  tile order. Useful wherever many mixed steps land at M=65-190, e.g. a smaller prefill budget or larger
  concurrency. Not worth an official run alone.
- **Lesson (worker):** `ws create --from-ws <built ws>` also needs `ws build`: T6-006 ran the baseline _C (KB61).

## Where headroom might remain (opinion)
- **Small mixed steps are partly launch-bound.** ~1500 extra eager launches per PIECEWISE mixed step: GDN chunk
  prefill + index/gather/elementwise glue, norms, attention.
  - Graphing or fusing the GDN prefill glue (T2's small item list) is an exploit item. Bounded by the
    un-profiled gap, <= ~15 ms per small mixed step: ~1-2% of the trial.
- **In-situ Marlin at M=49-96 is far above its cold cost.** The prefetch only covers the first GEMM, and GPU 0's
  clock is locked. Below M=48 there is no headroom we could reach cheaply.
