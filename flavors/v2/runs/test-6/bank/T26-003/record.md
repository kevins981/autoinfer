team:         T26
builds_on:    T26-002 (standalone microbench, no engine change)
performance:  "microbench: FINAL KILL. tuned swap-AB kernel gate_up chain P=32 99.1 vs Marlin 94.8 us (target <=85); qkvz 57.8 vs 43.3 (target <=39)"
summary:
  hypothesis: v1's loss (T26-002) was implementation: with 8 warps/SM, no spills, an 8-deep LDG.128 register ring and balanced split-K, the kernel reaches >=10% below Marlin in the chain arm.
  change:     agents/leader-26.w1/W1-3/swab3.cu: 256 threads, gate_up 198 regs / qkvz 142, no spills; CTA pairs split-K 2 (gate_up 17 tiles each), 256-K token chunks cp.async double-buffered; dequant bit-identical to Marlin.
  results:    Observed (GPU0, read flush, same job, mine/Marlin us): gate_up cold 109.6/105.1, warm 100.4/95.0, chain P=0 100.7/94.8, P=32 99.1/94.8; qkvz cold 59.2/51.5, warm 58.3/43.8, chain P=32 57.8/43.3.
              Ablations: gate_up loop-only 72.6, loads-only 97.9 cold / 90.4 warm; qkvz loop-only 46.1, loads-only 58.9/57.4. Loads-only rate in-kernel 0.71-0.99 TB/s vs 1.15-1.29 for the pure stream (T26-001 K1).
              Numerics: max|d| 4.9e-4, 0.10-0.15% elements differ, deterministic.
  conclusion: Interpretation: load/MMA overlap is now good (full ~ loads-only + 2..11 us), but the load side is the bound. Inside a real GEMM the weight stream drops to 0.7-1.0 TB/s and the L2 prefetch is barely cashed. Suspects, not isolated: per-CTA token staging (~63 MB of L2->SM traffic per GEMM, as much as qkvz's weights), barrier lockstep, and scattered per-warp streams. K0's 93% loop does not survive the real data path; Marlin stays ahead on warm/chain. Direction dead.
validity:     microbench only; numerics fp32 reorder only.
