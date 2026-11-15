team:         T1
builds_on:    T1-008
performance:  "not evaluated: microbenchmark only; folded into a smaller bundle on T2-008"
summary:
  hypothesis: humming W8A8 decode GEMMs run at ~72% of DRAM BW in situ (prof_t1008_node: 17.4 ms/step for 12.5 GB); tile tuning at M<=64 could recover ~3 ms/step.
  change:     (leader-1.w1) per-(N,K,M-bucket) tuning_config override table in vllm's humming integration (M<=64 only; VLLM_HUMMING_DECODE_TUNE=0 disables).
  results:    Observed: standalone cold-L2 sweep (jobs 66, 68; agents/leader-1.w1/hm/): defaults already at 84-88% of bytes/1008 GB/s on big shapes; best variants gate_up -2..-3.5% (stream-k off), down -2.6% (bk128), in_proj -4..-7%, 5120x3072 -7..-11% (bm32) but regime-dependent (L2-warm reverses in_proj / 3072 entries, job 70). Outputs differ by fp reordering only (max abs <= 0.031, one bf16 ulp). M>64 bitwise unchanged.
              In situ gap explained separately (prof_t1008_node): gate_up 105-110 us after attention layers vs 132-135 us after GDN layers (15,072 slow / 2,400 fast after GDN; 6,303 fast / 272 slow after attention): the GDN MTP kernel's per-position fp32 state writes drain from L2 into the next GEMMs (T2's K1 lane; T2-008 addresses it).
  conclusion: Interpretation: tile tuning is near the kernel family's ceiling (~0.15-0.5 ms/step best case); the "3 ms" napkin was wrong because the in-situ slowness is GDN writeback, not tiling.
validity:     fp reordering only in GEMM accumulation; generic by shape; not evaluated.
