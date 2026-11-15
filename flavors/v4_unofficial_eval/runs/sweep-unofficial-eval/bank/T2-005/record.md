team:         T2
builds_on:    T2-004
performance:  unofficial: PASS +2.30% vs T2-004 (unofficial #1); --long rerun cancelled when HEAD moved to T1-005; superseded by T2-006
summary:
  hypothesis: during each decode-step TP all-reduce (~60 us over PCIe) HBM is mostly idle; prefetching the first 32 MB of the next GEMM's FP8 Marlin weight into L2 (72 MB) on a side stream with 16 CTAs lets that GEMM read it from L2. Microbench with real NCCL (worker t2-w1, GPU job 19, agents/t2-w1/l2/sweep2.py, 8-layer chain, 5 interleaved rounds): 466 -> 428 us/layer (-8.3%) at 16 CTAs/32 MB; 62 CTAs instead slows the AR as much as it helps (job 11). Napkin end-to-end: ~2 ms/decode step, ~5%.
  change:     new vllm/model_executor/layers/l2_prefetch.py (triton streaming-load kernel; env VLLM_L2_PREFETCH_MB default 32, VLLM_L2_PREFETCH_CTAS default 16; 0 disables); Qwen3.5 decoder layer: around input_layernorm / post_attention_layernorm forward_fused (which perform the deferred TP all-reduce) fork a side stream that prefetches in_proj (qkvz/qkv) resp. gate_up weights, join after. Serve args = HEAD's.
  results:    Observed: unofficial #1 PASS +2.30% (B 706.7/712.1 vs ref mean 693.4), MTP acceptance 2.351 (unchanged), p99 TTFT 1112 / TPOT 42.0 ms. Server log on both ranks: 'L2 prefetch during TP all-reduce: 32 MB, 16 CTAs' (path active, captured in graphs without error). Microbench (job 19): -8.3% per layer chain.
  conclusion: Interpretation: the prefetch hides part of the next GEMM's weight read behind the all-reduce; end-to-end +2.3% is below the ~5% napkin (real layers have norms/GDN between, mixed steps gain little, AR slowdown). CTA count / MB not tuned in situ (16/32 from the microbench; 24 CTAs/40 MB was best there).
validity:     Scheduling only: the prefetch kernel only reads weight memory and writes a private sink buffer; no value used by the model changes (exact). Generic (not keyed to the benchmark); applies whenever TP>1 with the deferred-AR norm path. Memory: +256 KB sink, no change to KV/graphs sizes.

unofficial #1 (2026-11-15T06:14): PASS: +2.30% vs HEAD T2-004 (B 706.7 / 712.1 vs reference mean 693.4 over 2 starts, 0 run in this job; 200 prompts); p99 TTFT 1112 ms / TPOT 42.0 ms, MTP acceptance 2.351 (unofficial/1/unofficial.json)

unofficial #2 (2026-11-15T06:20): FAILED at exit 130 (unofficial/2/)
