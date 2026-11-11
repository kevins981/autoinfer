team:         T10
builds_on:    BASE (standalone microbench, no engine change)
performance:  "measurement only: e4m3 MMA runs 2x bf16, but its f32 accumulation keeps only ~13 bits below the leading term (C included); exact FP8-split GEMMs are impossible on sm_89"
summary:
  hypothesis: An exact Ozaki-style split of each bf16 activation into two FP8 pieces (hi e4m3 + lo e5m2/e4m3, exact
              by construction within a per-block range) feeds e4m3 x e4m3 mma.sync with fp32 accumulation. That runs at
              2x the bf16 rate and needs no weight dequant, which would lift the M=48 decode GEMM's tensor ceiling (Marlin
              68% of HMMA peak, KB66). Premise: e4m3 MMA rate = 2x bf16, AND its f32 accumulation is IEEE-fp32-like
              (products exact, then fp32 sums), as with bf16 HMMA.
  change:     tc.cu: rate kernels (bf16 m16n8k16 / e4m3 m16n8k32 f32-acc / e4m3 f16-acc, register-resident, 4-16 warps/SM),
              plus a 16x8xK precision check on identical values (e4m3 vs bf16 MMA vs fp64 reference). tc2.cu: one MMA with
              C = 2^p plus a single product 1.0 (and C = 1 plus product 2^p). Jobs 85, 87 (~0.0 GPU-min).
  results:    Observed:
              - Rate, TFLOP/s (GPU1 / GPU0): bf16 164.5 / 145.7; e4m3 f32-acc 329.0 / 291.3; e4m3 f16-acc 653 / 578.
                1 warp/scheduler already saturates in this register-only loop.
              - Precision on random e4m3 data, max |err| / sum|p|: e4m3 MMA 3.0e-4 (K=128), 5.0e-4 (K=1280), 3.5e-4 (K=5120).
                bf16 MMA on the same values: 2.9e-7 / 9.5e-7 / 1.6e-6. Bitwise-equal outputs: 0/128.
              - Single MMA: C=2^p plus product 1.0 is exact for p<=12 and LOSES the 1 for p=14..22. C=1 plus product 2^p:
                the same (exact p<=12, lost from p=14). (p=24 "exact" only because fp32 itself rounds 2^24+1.)
  conclusion: Interpretation: on sm_89, e4m3 mma.sync with ".f32" accumulation aligns all addends (the products AND C) to a
              ~13-14-bit window below the largest one. That is like what DeepSeek reported for Hopper FP8 WGMMA. So:
              (1) It is not fp32 accumulation: any FP8-MMA GEMM, including an exact hi/lo split of bf16 activations, drops
                  low-order bits. That is an approximation, not a reordering, on top of the run's ban on FP8 compute.
              (2) Even with promotion after every MMA (C=0), the error inside one k32 MMA is ~2^-13 of its largest product.
              Kills the "FP8 tensor cores with exact activation splitting" family (T4 rejected it on napkin math and range;
              this is the measured reason). Side facts: f16-acc FP8 is 4x bf16 but also lossy. The bf16 dense peak
              (146 TF on GPU0 at its locked 2235 MHz, 165 on GPU1) stays the exact tensor roof. INT8 needs >=4-6
              passes for exactness (T2), so it is no faster.
validity:     measurement only; no served change.
