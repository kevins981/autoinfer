## Goal
Serve **Qwen3.8-27B-FP8 on 2x RTX 4090** (no GPU peer-to-peer) faster: maximize **output tokens/s at concurrency 16** on a fixed ShareGPT benchmark (400 prompts, temperature 0), **without changing what the model computes**. FP8 weights and FP8 KV cache stay; activations stay bf16; speculative decoding must stay exact. Gates: TTFT p99 ≤ 1784 ms and TPOT p99 ≤ 58.1 ms (BASE +20%), GSM8K ≥ 89.6%, and one 128K-token request must still be served. Differences under 3% count as noise.

## Result
| | output tok/s | vs BASE | TTFT p99 | TPOT p99 | GSM8K |
|---|---|---|---|---|---|
| BASE (stock vLLM) | 631.4 | | 1487 ms | 48.4 ms | 92.8% |
| Round 1 champion T1-005 | 699.2 | +10.7% | 1309 ms | 50.6 ms | 93.2% |
| **Champion T1-009** (round 2) | **730.9** | **+15.8%** | 1438 ms | 41.3 ms | 93.6% |
| Best measured T1-012 (round 3) | 749.7 | +18.7% | 1426 ms | 43.3 ms | 93.6% |

**Decision for you:** T1-012 is +2.6% over T1-009, below the agreed 3% bar, so it is **not** the champion by rule. The gain is consistent across three measurements (same-job A/B +2.2%, microbenchmark ~2 ms/step saved, 2 official trials 746.0/753.5 vs 729.7/732.1), and the change is bitwise exact. I recommend promoting it by decision. Caveat: preemption/resume is untested for it.

## What worked
- **Host-staged all-reduce** (T1-002): the two GPUs exchange partial sums through pinned host memory instead of NCCL. Bitwise equal to NCCL; decode all-reduce 61 → 50 µs. +5.0%.
- **CUDA graphs for mixed prefill steps** (flag): worthless alone, but +4.8% on top of the faster all-reduce (T1-005).
- **Logits diet** (T1-006/007/009): local-argmax drafts, FP8 draft head (drafter only), and a greedy verification fast path that gathers (max, argmax) instead of full logits. Together +4.5%.
- **GDN state checkpoint + replay** (T1-012): write one recurrent state per decode step instead of one per speculative token. +2.6%.

## What didn't
- **3 draft tokens** (T1-008, T1-010, T1-011): +15% acceptance, but each request then needs more spec-state pages, only 14 of 16 requests fit, and no larger KV budget fits in memory.
- **NCCL tuning** (measured in a microbenchmark), **pipeline parallelism**, **Marlin retuning**, **reduced-vocab draft head**: ruled out with numbers.

## Things to know about this machine
- **GPU 0's clock is locked at 2235 MHz** (GPU 1 boosts to ~2600 MHz). GPU 0 is ~15% slower on compute, so ~7% of wall time is spent waiting for it. Accepted as fixed; BASE was measured under it too.
- No peer-to-peer between the GPUs: all GPU-to-GPU traffic crosses PCIe through host memory.

## Open questions / next steps
- **Promote T1-012?** (above)
- **R3-B v2**: shrink the per-request GDN spec state to one page so 3 draft tokens fit 16 requests in the same memory (est. a few % more; risky integration).
- Lossless compression of the all-reduce payload and L2 prefetch during the all-reduce: proposed, untried (see Untried ideas).
- Remaining decode step (~35 ms): Marlin FP8 GEMMs ~18.5 ms, host all-reduce ~5.7 ms, lm_heads ~3 ms, GDN ~3 ms.

## How to read this packet
- **Experiment tree**: every experiment, by direction. ★ marks the champion's path; click a node for what changed, why, the numbers, the conclusion, the validity review and the code diff.
- **Insights (KB)**: confirmed findings, each citing its records. **Untried ideas**: scored but not run. **Rules & assumptions**: the task definition, the human's rulings, and accepted facts.
- `diffs/` has each record's diff against the baseline vLLM; `data/` has the run's markdown files (reports, journal).
