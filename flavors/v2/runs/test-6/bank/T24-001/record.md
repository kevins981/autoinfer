team:         T24
builds_on:    T14-004 (standalone JIT microbench of T17-001's g17.cu with 3 m16 tiles (NM=3) at M=48; no engine change)
performance:  "microbench: premise KILLED. gate_up M=48 warm 114.8 us vs Marlin 85.8; no-B-loads 88.7, no-loads 80.4, B-loads-only warm 103.2"
summary:
  hypothesis: in-situ decode GEMMs sit at Marlin's L2-warm floor (KB79), which uses 49-67% of HMMA peak; a kernel built for
              M<=48 (no padding to 64) and fast when weights are in L2, plus prefetch, could cut decode GEMMs 15-25% (+7-8% e2e).
  change:     none to the engine. Worker leader-24.w1 ran T17's g17.cu at NM=3 (M=48), job 155, 0.2 GPU-min, read flush (KB76),
              warm = read flush + evict_last prefetch of min(48 MB, weights). Report: agents/leader-24.w1/W1-1/REPORT.md.
  results:    Observed (GPU0, M=48, profiler median of 15, us; Marlin cold/warm vs g17 cold/warm, no-B-loads, B-loads-only warm):
              gate_up 104.7/85.8 vs 127.6/114.8, 88.7, 103.2 (no loads at all 80.4 = 106 TF; B+dequant no MMA warm 106.5)
              qkvz 51.9/43.0 vs 63.7/59.5, 45.8, 51.4 | down 57.4/47.5 vs 76.5/69.6, 55.3, 59.9
              gdn_out 24.5/21.4 vs 39.7/37.8, 29.5, 31.7 | attn_qkv 47.2/40.1 vs 63.0/59.4, 45.9, 50.8. Chain ~= warm for both kernels.
              Exactness: max|d| 2.4-4.9e-4, 0.04-0.15% elements differ, deterministic (fp32 order only).
              My estimate before: gate_up no-B-loads ~66-72, warm ~72-76. Gates: no-B-loads <= 75 and B-loads-only warm <= 70: both fail.
  conclusion: Interpretation: at M=48 the MMA is not the limiter: M 64->48 cuts MMA work 25% but no-loads time only ~10% (88.6->80.4);
              the per-B-fragment fixed cost (dequant ALU, smem round trip, barrier, ldmatrix) dominates, and even with NO loads the
              kernel (80.4) is within 6% of Marlin's full warm kernel (85.8). A cp.async ring can't feed from L2 at Marlin's rate
              (qkvz loads-only warm 51.4 > Marlin full 43.0). Marlin is close to a realistic warm floor at M=48 (agrees with KB39,
              KB49, KB66/68); a warm-optimised GEMM + prefetch has <= ~5% of the GEMM to win (~+2% e2e). Untuned (grid/NST/warps),
              48 MB prefetch covers only half of gate_up; other shapes show the same ordering.
validity:     microbench only; numerics fp32 reorder only. No served change.
