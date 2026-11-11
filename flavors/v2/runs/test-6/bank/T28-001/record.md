team:         T28
builds_on:    T27-002 (microbench only; no engine change)
performance:  "microbench: custom small-M FP8 W8A8 kernel v1 slower than Marlin and Triton at M>=48 (M=64 total 22.48 vs Marlin M=48 16.47 ms cold); stock Triton M=64 16.63 ~ Marlin M=48 16.47 (cold), 14.84 vs 13.75 warm"
summary:
  hypothesis: an FP8 W8A8 GEMM at M=64 (e4m3 mma, fp32 rescale per 128-K block, standard [N,K] layout) is byte-bound and costs about Marlin's M=48 time, removing the k=3 cliff (KB63).
  change:     agents/leader-28.w1/k/w8a8_small.cu (swap-AB, register-direct weights: one warp per 16-row weight tile, LDG.128 -> mma A fragments; activations as the B operand from L1/L2; split-K + reduce; ring depth D in {2,3,4}); bench agents/leader-28.w1/mb/bench.py (T29's harness, real weights, 256 MB READ flush, CUDA graph x8). Job 193 (0.4 GPU-min, GPU0).
  results:    Observed: relF vs fp32 block-scaled reference 1.67e-3 for both the custom kernel and Triton (identical on every shape/M).
    Per-forward linear totals ms (cold | warm): M=16 Marlin 15.23|10.35, Triton 15.33|11.77, mine 15.29|14.61; M=48 16.47|13.75, 15.97|13.40, 20.17|16.32;
    M=64 17.81|16.13, 16.63|14.84, 22.48|17.55; M=80 24.01|21.03, 21.16|19.21, 25.80|19.14; M=96 24.17|21.15, 21.29|19.32, 26.09|19.41.
    gate_up cold us at M=64: Marlin 124 / Triton 116 / mine 160 (0.56 TB/s); custom kernel time grows ~linearly with M (gate_up 106/143/160/179/184 us at M=16..96) while Marlin/Triton barely move 16->64. Warm M=96 gate_up: mine 136 vs Marlin 151 / Triton 161. ptxas 180-255 regs.
  conclusion: Interpretation (worker, no profile): v1 is issue/LSU-bound (activation LDG.128 per warp 6x the weight loads at M=96, strided As loads, scale FMAs), occupancy ~12 warps/SM. Not the MMA (tensor floor ~52 us) nor split-K. Fixes: stage activations+As in smem per CTA, 32 rows per warp, column-major As, cp.async weight ring.
    Leader: the useful measured fact is that STOCK Triton already puts M=64 at ~Marlin's M=48 cold (+0.16 ms/forward) and +1.1 ms warm; the custom kernel's job is now the warm/in-situ arm (and M=48 itself), worth <= ~1 ms/step.
validity:     microbench only, nothing to review. Math per the Activations ruling (fp8 MMA acc within a 128-K block, fp32 across blocks).
