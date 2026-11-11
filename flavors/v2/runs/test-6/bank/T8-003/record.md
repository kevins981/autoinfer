team:         T8
builds_on:    T1-042 (standalone JIT kernel on the champion's Marlin weights; no engine change)
performance:  "microbench: M1 KILLED. 2-warps/scheduler FP8 decode GEMM, gate_up M=48 warm 129.5 / cold 131.7 us vs Marlin 87.8 / 95.0 (gate warm <= 72)"
summary:
  hypothesis: a 2-warps/scheduler M<=48 kernel turns the 91.5% inner-loop rate of T8-002 into warm gate_up <= 72 us (Marlin 84-88).
  change:     agents/leader-8.w1/g2w.cu (copy here) + bench.py/abl.py/prof.py. CTA = 8 warps (2 n32 halves x 4 k-warps), grid 128,
              K split in 8 slices of 640 with A fragments register-resident, B via 5-stage cp.async ring (8 KB/stage), reads
              Marlin's repacked B + scales unchanged, dequant bit-identical to Marlin; k-warp reduce via smem, cross-slice reduce by an
              owner CTA (release/acquire counter), deterministic. Worker leader-8.w1, jobs 63-77, ~1.9 GPU-min.
  results:    Observed (GPU0, CUDA graph chain read-flush or L2pf 88 MB -> gate_up -> down-sized Marlin, median of 15, job 77):
              M=48 Marlin 95.0 cold / 87.8 warm; mine 131.7 / 129.5. M=32 91.5/83.0 vs 117.9/113.7. M=16 89.1/78.3 vs 106.7/101.2.
              v1 (last-arriver reduce) 185.6 warm. Exactness vs Marlin: 0.06-0.13% elements differ, max abs 4.9e-4 (|y|<=0.2),
              >1 ulp only near-zero outputs (0.004-0.008%); run-to-run bitwise.
              Ablations: loop alone (no cp.async, no epilogue) 74 us = 115 TF = 79% of peak (math_pipe_throttle top stall), not 91.5%.
              With the B stream: 1.39 us/stage vs 0.87 MMA-only; achieved B rate 0.75 TB/s (L2 can give 4.2); owner reduce tail ~9 us,
              startup ~8 us (LSU-throttled A ldg); stalls barrier 5.7, math 5.6. Marlin cold 95.0 us = at the HBM read roof for 89 MB.
  conclusion: Interpretation: the 2-warp loop keeps ~79% (not 91.5%) inside a realistic CTA, and the load path (strided 1 KB pieces,
              lockstep barrier per stage) loses far more; Marlin's cold path is at the roof and its warm path is hard to beat.
              Even at 79% with a perfect load path, warm gate_up >= ~74 us: decode gain <= ~1.3 ms/step (~+2.8%), below the 5% bar.
              Direction T8-1 dead as specified. Correction to T8-002's reading: its 91.5% was a toy loop (12 MMAs/warp, no stream).
validity:     microbench only; numerics fine (fp32 reorder only).
