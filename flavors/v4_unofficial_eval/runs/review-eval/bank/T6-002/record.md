team:         T6
builds_on:    T3-007
performance:  "INVALID (double all-reduce bug); stopped after 1 start: 844.3 tok/s (eval/1, GPU job 109, STOP_BELOW=853.6), acceptance 2.294 (champion 2.337)."
summary:
  hypothesis: in mixed steps, run each row-parallel GEMM that feeds an all-reduce (GDN out_proj, attention o_proj, MLP down_proj) in 2 row halves.
              Half 1's NCCL all-reduce, on a persistent side stream, overlaps half 2's GEMM (framework 06: Window over Split r; T5's #2, handed to T6).
              Napkin after review T6-1: +0.5-2%.
  change:     new vllm/distributed/device_communicators/t6_split_ar.py; a RowParallelLinear.forward hook (M >= VLLM_T6_SPLIT_MIN_ROWS=192, TP=2);
              t6_split flag on the 3 linears; Qwen3_5RMSNorm.forward_fused skips its AR when the input is the split's output (by leader-6.w1,
              decorator placement fixed by leader-6). HMAR as in the champion.
  results:    Observed: 844.3 tok/s, 1 start; acceptance 2.294; gate_latency pass. Worker A/B, quick, separate jobs, same buggy code + HMAR768:
              ON 858.5 (job 107, acceptance 2.30) vs OFF 843.4 (job 105, acceptance 2.34); median mixed-step elapsed -8..-10% at 192-767 rows, flat at 768+.
  conclusion: Interpretation: the code is wrong. The target's and the MTP model's final norms (fused_allreduce_gemma_rms_norm) and the aux hidden-state
              capture all-reduce again an output the split had already reduced (residual + 2*s on qualifying steps). Found by leader-6 from the acceptance
              drop and confirmed by review reviews/T6-2. The throughput numbers are not usable. Fixed in T6-003.
validity:     INVALID as filed (changes the computed function on mixed steps >= 192 rows).
