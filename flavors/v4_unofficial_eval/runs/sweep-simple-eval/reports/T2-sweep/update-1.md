# T2 map update 1 (resync 1)

## Tested
| id | what | result | evidence |
|---|---|---|---|
| W1 | prefill/mixed CUDA graphs to 2048 tokens (flag) | **win, promoted (T2-001)**: 648.6 / 658.6 vs BASE 634.5 / 643.4 (+2.3%), 128K pass. Napkin said ~+13%: the graphs removed the cross-rank AR wait in mixed steps (NCCL fixed part 40 -> 7.7 ms), but mixed steps are "breakable" graphs: the GDN chunked prefill core (~1,700 eager Triton launches) and prefill attention still run eager between ~40 graph pieces | job 4; profile agents/leader-2/prof_cg2048 (T2-001 record) |
| R4 + draft local argmax | FP8 draft lm_head copy + use_local_argmax_reduction (flags) | first try OOM in graph capture (extra 0.64 GB); with KV 4 GiB: 645.3, **stopped after 1 start**, acceptance unchanged (2.346), TPOT p99 65 ms (unexplained) | T2-002 |
| (T1) NCCL_PROTO=Simple on T2-001 | | win, T1-002 promoted (+1.0%) | T1-002, built on my worker's AR table (leader-2.w1 jobs 2/5) |
| (T1) MTP k=3 | | 602.5 after 1 start; acceptance 2.701; step +22% | T1-001 |

## Verdicts changed (with evidence: agents/leader-2.w1/gemm_results.md, job 16)
- 5-compare disagreement 2 settled: FP8 MMA on the 4090 runs at ~310-320 TFLOP/s (torch._scaled_mm, M=2048);
  vLLM's Triton block-FP8 kernel reaches 240 (default config, untuned), Marlin 150-160. **R6 (W8A8 for
  mixed steps) stays a candidate**: Triton block is 1.3-1.45x Marlin at M=256-512. Blocker: one weight
  layout must serve decode (Marlin) and prefill; investigating humming's FP8-activation path (worker).
- **R8 (Marlin decode efficiency) downgraded**: isolated Marlin at M=48 already streams 79-83% of peak on
  the 3 big shapes (out_proj 65%); the in-situ 68% loss is between kernels, not inside them.
  Remaining per-kernel headroom ~1-2 ms/step, not 4.5.
- 5-compare disagreement 1 settled as "both": T1 paired the AR kernels (mixed-step AR = ~300 us transfer +
  ~390 us wait); my graphs removed the wait, T1's Simple protocol cut the transfer.
- New: mixed steps' eager GDN prefill core is the remaining Host-resource cost (W1b): capture the GDN
  chunked prefill inside the graph (make it graph-safe), or cut its launch count.

## Current ranking (vs champion T1-002, ~660 tok/s)
1. R5 reduced-vocab draft head, **bf16 slice, no FP8 kernel**: a draft-only head of the first N=98K token
   ids (covers 99.6% of generic text tokens: vLLM docs, 598K tokens; data-free choice = tokenizer id order),
   split evenly across the 2 ranks (0.5 GB/GPU bf16, fits with KV 4 GiB) -> 2 x (1.37 -> ~0.55 ms) = ~1.6 ms/step (~4%).
   Exact: verification unchanged.
2. R6 W8A8 for mixed steps if one weight layout can serve both: ~3-5%.
3. W1b graph-capture the GDN prefill core: unknown size; needs a TRACE=graph profile to price.
4. K1 GDN rollback-free commit: 1-2 ms/step (2-3%); ncu check first.
5. S1 MTP k=3 after 1 and T1's faster AR.
