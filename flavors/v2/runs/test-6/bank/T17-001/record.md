team:         T17
builds_on:    T14-004 (standalone JIT kernel on the champion's Marlin FP8 layout; no engine change)
performance:  "microbench: M0 KILLED. Stream-K FP8xbf16 GEMM, gate_up M=64 125.8 us (CTA end; profiler 134) vs Marlin 103.5 (read flush); M=48 116.6 vs 92.5"
summary:
  hypothesis: Marlin's 3->4 m-block step (M=49-64) is ~39% above max(HBM, tensor) (T6-001), so a kernel overlapping stream and math
              makes M=64 cost ~M=48 and k=3 speculation pays (proposals/T17-1.md).
  change:     code/g17.cu: 128 CTAs x 16 warps, stream-K, 256-col tile, 3-4 stage cp.async ring (64-k stages), 4 m16 tiles per
              dequantized B fragment, Marlin's repacked layout + scales, dequant bit-identical to Marlin, fp32 acc, deterministic fixup.
              Worker leader-17.w1; jobs 126, 127, 129-131, 133; ~0.7 GPU-min (ws t17-k from T14-004, built).
  results:    Observed (GPU0, gate_up 17408x5120 per rank):
    - Flush method: 256 MB WRITE flush + graph-minus-flush (T5-002/T6-001 method) gives Marlin cold M=48/64 110.6/124.1 us
      (reproduces T5-002 111.2/126.3); a 256 MB READ flush gives 92.5/103.5. Dirty write-backs of the flush land in the GEMM (+18-20 us).
      With the read flush Marlin M=48 = 89 MB / 0.96 TB/s = at the HBM roof; M=64 is 1.11x roof (not 1.39x).
    - Mine M=64: math only (no cp.async) 88.6 us = 130 TF (89% of 146 TF peak); B stream alone 106 us = 0.84 TB/s (T8: 0.75);
      A loads ~free; NST 3 vs 4 same; all together 119 us compute-done + stream-K fixup 6-7 us median (15 max) -> 125.8 median end.
    - Chain with 45 us spin + 40 MB evict_last prefetch on a 2nd stream: M=64 133.9 vs Marlin 102.7; M=48 112.8 vs 86.0.
    - Exactness vs Marlin at M=1/16/32/48/64: max|d| 4.9e-4 (|y|<=0.19), 0.12-0.16% elements differ, >1 ulp only 0.006-0.011% (near-zero outputs); bitwise run to run.
  conclusion: Interpretation: the premise was a flush artifact. Marlin is at the HBM roof at M=48 and only ~11% above it at M=64.
    At M=64 gate_up math alone needs ~89 us vs a 93 us HBM roof, so even a perfect load path lands at ~97-100 us incl. tail:
    recovers <= 1/3-1/2 of the 48->64 step; k=3 payoff ceiling ~+2-3%, below the bar. Untested levers: barrier-free pipeline,
    2 CTAs/SM, B access order across CTAs (none can beat max(89,93)+tail). Full table: M0_SUMMARY.md.
    Process KB: "cold" Marlin numbers taken with a write flush include ~20 us/gate_up of flush write-back; use a read flush for roof comparisons.
validity:     microbench only; numerics fp32 reorder only.
