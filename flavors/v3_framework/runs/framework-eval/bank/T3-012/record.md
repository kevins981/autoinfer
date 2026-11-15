team:         T3
builds_on:    T3-009
performance:  unofficial: within noise / slightly negative (-0.38%, ABBA job 134; all B < all A but B ~= A's second start)
summary:
  hypothesis: The gate_up weight is 89 MB/rank; prefetching 56 MB instead of 40 during the AR window may serve more of it
              from L2.
  change:     env only: B = VLLM_AR_PREFETCH_MB=56 (affects gate_up only; qkvz 42 MB / qkv 37 MB are capped by size).
  results:    Observed: A 911.5/904.9 vs B 904.7/904.8 tok/s (-0.38%); acceptance 2.330-2.354.
  conclusion: Interpretation: the AR window (~44 us) can't pull much more than ~40 MB; extra lines either arrive after the
              GEMM starts or evict useful ones. 40 MB stays.
validity:     env-only, exact.
