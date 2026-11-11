team:         T1
builds_on:    T1-020
performance:  "microbench only: Marlin roofline + new instantiations; best variant projects ~0.28 ms/step -> NO-GO"
summary:
  hypothesis: Marlin decode GEMMs (~13-15 ms/step) have headroom vs the read roofline, esp. when L2-warm from our prefetch.
  change:     microbench + sweep only (sweep-only env overrides in ws leader-2.marlin2 / leader-2.marlin.sweep; not a candidate).
              Worker leader-2.w10, agents/leader-2.w10/ (NOTES.md, mb/r1..r4, mb/ncu).
  results:    Observed (GPU 0, 2235 MHz, M=48, real shapes, CUDA graph): cold Marlin at 86-92% of a pure-read roof for the big
              shapes (gate_up 106 us vs 97.6 read), 73% for the 16 MB out_proj/o_proj (~6.5 us fixed). Per step (target): all cold
              15.0 ms, as-prefetched 12.8 ms, cold read roof 13.2. Warm gate_up 84.4 vs ~57 floor (L2 read = MMA floor).
              Config sweep (all compiled tiles, SM counts): ~0 at M=48; M=16 128x64 beats default 128x128 by 2-4 us/GEMM (~12 us/step).
              ncu: tensor-pipe limited (math_pipe_throttle top stall; 1 warp/scheduler, 128-thread blocks; 52-54% tensor peak cold,
              64% fully L2-resident); slice reduction/locks minor. New instantiations: 256-thread 64x128/128x64 impossible
              (b_sh_wr_iters=1); 128x128 3-stage 8-warp: -2..-3.6 us warm, +0..1.6 cold -> ~0.28 ms/step best case; odd stage
              counts on 64x128 give wrong outputs/crash. Estimate (roofline) up to ~3.2 ms/step warm; achievable by config ~0.28.
  conclusion: Interpretation: cold Marlin is near the HBM roof; warm Marlin is tensor-pipe bound (bf16 x fp32-acc half rate on
              GeForce, one warp per scheduler). Real headroom needs a restructured kernel (2 warps/scheduler): ~1-1.5 ms/step max.
validity:     measurement only.

Follow-up (Triton W8A16 probe, agents/leader-2.w10/tr, jobs 80-81, 1.0 GPU-min): Marlin's exact dequant (bf16_rn(fp8 x s_bf16)) matched,
max diff 1 ulp. Best Triton config (warps 4/8, stages 3-5, split-K 1/2/4, swap-AB) at M=48 cold/warm: gate_up 175/143.5 vs Marlin
107.4/83.9; down 95/74.7 vs 56.8/45.8; in_proj 65.5/60.9 vs 52.7/42.2. At M=16 parity. Off-the-shelf generator doesn't beat Marlin;
only a hand restructure (2 warps/scheduler) could, multi-day for <= ~1-1.5 ms/step warm-only.
