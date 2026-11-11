team:         T8
builds_on:    BASE (standalone microbench, no engine change)
performance:  "measurement only: Marlin's warm M=48 ceiling (~68% of tensor peak) is the 1-warp/scheduler latency limit; the same inner loop with 2 warps/scheduler reaches 91.5%"
summary:
  hypothesis: KB28's "warm Marlin is tensor-pipe bound at ~64% of peak" is a latency limit of 1 warp per scheduler, not the
              hardware; T1-029's 2-warp prototype failed on its load path, not on the math loop.
  change:     tc.cu (this folder). P1: HMMA m16n8k16 bf16->fp32 with register operands. P2: Marlin-like inner loop from SMEM
              at M=48: ldmatrix.x4 A (3 m16 tiles), one 32-bit FP8 B word per n8 tile per lane -> bf16 by shift/mask, x scale
              (HMUL2.BF16), 3 mma per n8 tile, 4 n8 tiles per warp. P3: cp.async.cg 8-stage ring from an L2-resident 32 MB
              buffer vs a cold 1 GB buffer. Job 62 (~0.05 GPU-min).
  results:    Observed (TFLOP/s; GPU0 locked 2235 MHz / GPU1 2520 MHz):
              P1 regs-only: 145.6-145.9 / 164.6-165.1 at 1, 2, 4 warps/scheduler (= the 165.2 TF spec scaled by clock).
              P2 smem+dequant+HMMA M=48: 1 warp/sched 99.8 (68.4%) / 112.8; 2 warps/sched 133.5 (91.5%) / 151.1.
              P3 cp.async: L2-resident 4175-4199 / 4619-4626 GB/s; HBM 953-955 GB/s (1 or 2 CTA/SM).
              Cross-check: T1-022 warm gate_up 84.4 us = 8.55 GFLOP / 84.4 us = 101 TF, i.e. P2's 1-warp number.
  conclusion: Interpretation: the warm decode GEMM gap is software (one warp per scheduler cannot keep the HMMA pipe fed
              through ldmatrix + dequant latency), and the L2 path is >4x faster than needed. A 2-warps/scheduler M<=48 kernel
              could run warm GEMMs ~1.34x faster (gate_up ~64 vs 84 us). Not yet shown: a full kernel (multistage cp.async,
              group scales, split-K reduction, epilogue) keeping most of the 91.5%; cold path at >= Marlin's 86-92% of HBM.
              Note: T1-022's 8-warp Marlin config (128x128, 3-stage) gained only 2-3.6 us warm. Marlin's warp split
              has far fewer independent MMAs per warp than P2's 12, so it is not a counterexample, but it is the main risk.
validity:     measurement only.
