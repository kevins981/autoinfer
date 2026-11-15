team:         T3
builds_on:    T3-009 + T2's prefetch-aware GEMM configs (ws t2-mm, leader-2.w1)
performance:  unofficial: within noise (-0.03%, ABBA job 135, mixed)
summary:
  hypothesis: T2 found (agents/leader-2.w1/gemm/) that with 40 MB prefetched, split-K / larger-BN Triton configs are much
              faster at M=48 (in_proj -24%, qkv -13%, gate_up -9% micro); T2's server smoke (-0.8%, 1 pair) ran on T2-009
              code where the in_proj prefetch was never delivered, so it was retested on T3-009.
  change:     (worker leader-3.w1, ws t3-mm) T3-009 + t2-mm's fp8_utils.py (gate VLLM_T2_GEMM_PF_CFG; split-K with
              per-split fp32 partials + fixed-order reduce kernel: deterministic, no atomics).
  results:    Observed: ABBAAB, A gate 0, B gate 1: A 902.8/892.3/913.9 vs B 899.0/903.3/905.8 (means 903.0 vs 902.7).
  conclusion: Interpretation: the microbench gain doesn't appear in situ; the split-K partial+reduce launches and writes
              likely eat it. Dropped.
validity:     fp32 summation order across splits only; deterministic. Not proposed.
