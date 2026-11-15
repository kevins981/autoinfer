team:         T2
builds_on:    T1-005
performance:  unofficial negative (1 start): -1.82% vs HEAD T1-005
summary:
  hypothesis: during each decode-step TP all-reduce (~60 us over PCIe) HBM is mostly idle; prefetching the first 32 MB of the next GEMM's FP8 Marlin weight into L2 (72 MB) on a side stream with 16 CTAs lets that GEMM read it from L2. Microbench with real NCCL (worker t2-w1, GPU job 19, agents/t2-w1/l2/sweep2.py, 8-layer chain, 5 interleaved rounds): 466 -> 428 us/layer (-8.3%) at 16 CTAs/32 MB; 62 CTAs instead slows the AR as much as it helps (job 11). Napkin end-to-end: ~2 ms/decode step, ~5%.
  change:     T2-005 rebased onto T1-005 (host-pinned AR); new vllm/model_executor/layers/l2_prefetch.py (triton streaming-load kernel; env VLLM_L2_PREFETCH_MB default 32, VLLM_L2_PREFETCH_CTAS default 16; 0 disables); Qwen3.5 decoder layer: around input_layernorm / post_attention_layernorm forward_fused (which perform the deferred TP all-reduce) fork a side stream that prefetches in_proj (qkvz/qkv) resp. gate_up weights, join after. Serve args = HEAD's.
  results:    Observed: unofficial #1 negative (1 start) -1.82% (715.3 vs 728.6). On T2-004 (NCCL AR) the same change was +2.30% (T2-005).
  conclusion: Interpretation: T1-005's host one-shot AR is shorter (less window for the 32 MB prefetch) and probably more sensitive to concurrent HBM/SM traffic; the 32 MB/16 CTA setting tuned for NCCL no longer fits. One start only: a screen negative, not a measured fact.
validity:     Scheduling only: the prefetch kernel only reads weight memory and writes a private sink buffer; no value used by the model changes (exact). Generic (not keyed to the benchmark); applies whenever TP>1 with the deferred-AR norm path. Memory: +256 KB sink, no change to KV/graphs sizes.



unofficial #1 (2026-11-15T06:34): negative (1 start): -1.82% vs HEAD T1-005 (B 715.3 vs reference mean 728.6 over 2 starts, 0 run in this job; 200 prompts); GSM8K/128K not run (stopped after 1 start), p99 TTFT 1259 ms / TPOT 38.7 ms, MTP acceptance 2.358 (unofficial/1/unofficial.json)
