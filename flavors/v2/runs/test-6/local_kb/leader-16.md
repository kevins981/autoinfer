# leader-16 (T16, explore) notes
## 2026-11-09 start
- Read KB1-73, writeups T2-T15. Consensus: decode step pinned by coupled HBM+bf16 tensor floors; prefill at bf16 roof; exact FP8/INT8/FFMA dead.
- Largest non-floor bucket: decode AR. Ceiling of removing an AR = 21-25 us (T9-006 A_pf-N), x128/step ~2.9 ms of 23.8 -> ~+8% e2e ceiling.
- Hybrid CE-write + SM-read transport floor 20.7 us/384KB (T15-002) vs hostar ~41-46. Unknown: CE start latency in graph (T9-006 all-CE lost 23 us/AR).
- Other package candidates: GPU1 MLP rebalance in mixed steps using memory freed by route B (T5-005) (~1.5-2.5%); drafter collectives via same AR (~0.5%); lm_head screen (~0.8%).
- W1-1 (leader-16.w1, ws t16 = T9-005 built): hybrid AR microbench, gate >=10 us/AR better than A_pf in-chain. Budget 6 GPU-min.
- Manager: the GPU1 MLP rebalance (duplicated FP8 slices) needs a ruling first: MB per rank, memory source, 16-request admission, 128K test. Two-slot GDN must be ported to T9-005 and re-verified. Compare AR arms by payload (raw vs codec).
- New champion T14-004 956.6 (T6 Marlin slab policy for M>48, mixed steps only). Bar 1004.4. Decode AR untouched -> hybrid AR stacks; rebase proposal forecasts on 956.6.
- T16-001: hybrid AR FAILED. Best H1pf_2 is 11 us/AR slower than A_pf; standalone 68 vs 38.5 us. The CE branch in a graph has ~20 us first-piece latency. Decode AR line closed. Decision: no bar-clearing direction left with decent odds -> recommend voluntary close.
