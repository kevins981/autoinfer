team:         T26
builds_on:    BASE (standalone microbench, no engine change)
performance:  "microbench: K0 GO (no-loads swap-AB loop 62.9 us = 93% HMMA peak on gate_up M=48 work, vs 73-79% for T8/T17/T24 designs); K1 cold stream 0.95 TB/s (HBM roof)"
summary:
  hypothesis: putting FP8 weights on the MMA A side, loaded global->registers (no smem/barrier on the weight path), with tokens on n8 (6 MMAs per dequantized fragment), removes the per-fragment cost that held earlier decode GEMMs at 73-79% of HMMA peak (KB81).
  change:     agents/leader-26.w1/W1-1/sw.cu (copy here). K0: inner loop only (Marlin-identical dequant+scale on changing regs, ldmatrix token frags, barrier/16 ksteps). K1: weight stream only, ld.global.nc.L1::no_allocate.v4.
  results:    Observed (job 165, GPU0 2235 MHz): K0 62.9-64.1 us over W in {8,12,16} warps/SM x {1,2,4} tiles/warp (91.6-93.4% of 58.7 us peak); no-dequant 62.0-63.2 (dequant ~1 us). Regs 66-139, spills only W16/T4.
              K1 read-flush cold: gate_up 89 MB 92.4 us (0.95 TB/s), qkvz 36.5 us, out_proj 13.5 us; warm: qkvz 32.5 us (1.29 TB/s, below the 1.5 gate, cause unknown), out_proj 3.7 TB/s, gate_up w/ 48 MB pf 80 us. Flat across in-flight 8-64 KB/SM.
              Estimate before: K0 <=66 us GO line; K1 cold >=0.93, warm >=1.5 TB/s.
  conclusion: Interpretation: the 73-79% ceiling of earlier kernels was their smem-staged weight path, not dequant or HMMA issue. Open question: whether loads and math overlap in one kernel (K2, T26-002).
validity:     microbench only; no served change.
