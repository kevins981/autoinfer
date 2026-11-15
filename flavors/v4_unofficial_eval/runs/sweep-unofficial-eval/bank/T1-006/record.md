team:         T1
builds_on:    T1-005
performance:  screen: no gain (1 start per arm, job 37); not unofficial-evaluated
summary:
  hypothesis: overlap the row-parallel all-reduces of large (prefill-containing) steps with the GEMM by splitting rows into K=2 chunks (AR of chunk 1 on a side stream while chunk 2 computes). Microbench (t1-w1 job 20): NCCL AR on a side stream hides 74% behind bf16 GEMMs. Napkin after the rank-skew analysis (half of prefill AR time is the faster rank waiting, t1-w1 skew.py/skew2.py on baseline nsys): 2.5-5% before T2-004, less after (graphed mixed steps <=512 tokens).
  change:     RowParallelLinear._chunked_matmul_all_reduce (env VLLM_CHUNKED_AR_MIN_ROWS default 0=off, CHUNKS, CHECK, GRAPH); Qwen3.5 model: when enabled, turn off use_fused_allreduce_norm so the AR runs in the linear (the model defers the AR into the next norm otherwise: first prototype was dead code).
  results:    Observed (job 37, 100 prompts c=16, T1-005 args, 1 fresh start each): off 711.3 tok/s; chunked eager-only (rows>=192, effectively steps >512 tokens) 711.3; chunked also inside CUDA graphs 702.0 (-1.3%, p99 TPOT +9%). Per-layer check at 2048 rows: bitwise equal (80 checks). Acceptance unchanged.
  conclusion: Interpretation: no visible gain; the applicable steps are few after T2-004, half of the in-step AR time is rank skew (GPU0 runs compute-bound Marlin ~12% slower), side-stream NCCL slows the concurrent GEMM, and K=2 halves Marlin's efficiency. Screen negative (1 start per arm), dropped. Not for promotion.
validity:     exact (bitwise equal per layer at 2048 rows); off by default.
