team:         T3
builds_on:    T3-009 (profile only)
performance:  analysis: k=3 step +29% is mostly fixable implementation (GEMM M-tiling at 56/64 rows; DeltaLog 2-row cap)
summary:
  hypothesis: (spar T3-1) T2-013's k=3 loss (step +32% vs a +12% napkin, TTFT p99 doubled) may be implementation, not
              inherent.
  change:     none: nsys of T3-009 code, num_speculative_tokens 2 vs 3, c=16, 56 prompts (worker leader-3.w1, job 142;
              data agents/leader-3.w1/prof8_k2, prof8_k3, kcmp*.py).
  results:    Observed: steady decode step 27.7-28.1 ms (k=2, 48 rows) vs 35.8 ms (k=3, 14 decoders = 56 rows; +29%).
              Kernel ms/step k2 -> k3: GEMM 15.4 -> 20.2 (+4.8: gate_up 107.7 -> 128.3 us, qkvz 40.0 -> 62.7, qkv 35.4 ->
              58.8, out/down 36.7 -> 41.2; grids +33%: M=56 uses the M=48 BLOCK_M=16 config -> 4 M-tiles instead of 3),
              GDN decode 1.9 -> 4.5 (+2.6: DeltaLog caps at 2 logged rows, so 4-row steps write all states: 36.7 -> 87.9
              us/call), hostmem AR 6.6 -> 8.1 (+1.6), other 2.4 -> 3.3 (+1.0, extra draft forward). Same graph launches
              (3 cudaGraphLaunch + ~47 eager kernels per step) in both; idle ~2%. Acceptance 2.42 -> 2.83 (+17%). Mixed steps
              similar per bucket; TTFT mean 353 -> 734 ms, cause not pinned (scheduling of prefill order + longer steps).
  conclusion: Interpretation: with M=56/64 GEMM tiling fixed and DeltaLog extended to 3 rows, the k=3 step would be ~28.4
              ms (+3% vs k=2) for +17% tokens/step: a candidate. TTFT must be re-checked (gate).
validity:     profile only.
