team:         T1
builds_on:    T1-042
performance:  "unofficial: KILLED before building: upper bound 1.8 us/layer at M=48 (kill line 3)"
summary:
  hypothesis: The GDN ba projection (cuBLAS bf16 Kernel2 + splitK, 7.7 us x 48) is far from its ~1 us floor; one skinny kernel would save 0.2-0.25 ms/step (anatomy row "small bf16 GEMM").
  change:     none (ws leader-2.baproj unmodified, T1-042 code). Overlap microbench out/mb_overlap.py.
  results:    Observed:
              - In-server, ba runs on a second stream concurrently with the qkvz Marlin (execute_in_parallel). At 16 requests it finishes before Marlin: Kernel2 544.9-549.1, splitK end 587.5, Marlin 545.2-591.1 us.
              - Job 182 (cold, CUDA graph, aux stream as in-server), us/layer at M=48: Marlin alone 52.0, overlap 53.8, serial 57.3, ba alone 4.9 -> upper bound 1.8 us/layer (~0.09 ms/step). M=16: 0.7. M=64: 2.6.
              - At M=3/6 (1-2 requests), cuBLAS picks a slow kernel (53.5 us vs Marlin's 38.5 in-server), exposing ~15 us/layer on tail steps only. Estimated <= 0.1% on this benchmark.
  conclusion: Interpretation: the anatomy's 0.35 ms "small bf16 GEMM" was overlapped kernel time, not critical path; the attribution split concurrent time. Killed.
validity:     N/A (no change).
