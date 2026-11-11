team:         T6
builds_on:    T1-042
performance:  "microbench only: Marlin per-step -3..-8.6 ms at M=72-120, M=176 (12-26%); M=49-64 unchanged; M=144 +1.4 ms. Gate G1 FAILED."
summary:
  hypothesis: running all rows of a GEMM as parallel slabs in ONE Marlin launch (no remainder launch) lets slabs share weights via L2 => M=65-127 near roof.
  change:     marlin.cu host: M>64 -> S=ceil(M/64) slabs of round16(ceil(M/S)) rows in one launch (env AI_SLAB, default on; AI_SLAB_S49 for M=49-64). marlin_template.h: per-slab prob_m, on-the-fly A predicate, C-reduce masks for short slabs. (ws t6-slab, built; leader-6.w1)
  results:    Observed (jobs 33, 34; GPU0 cold L2-flushed per-step sum; stock/new ms): M=48 15.57/15.58; 64 17.10/17.11; 72 24.86/21.76; 80 25.27/21.76; 96 27.50/21.94; 112 29.15/24.91; 120 33.59/25.00; 128 25.11/25.06; 144 33.03/34.39; 160 35.36/34.41; 176 37.98/34.39; 192 34.56/34.55. M=49-64 as 2x32: 21.0, 4x16: 29.6.
    Per shape: gate_up (89 MB) 188-196 us for any 2-slab M (72..128), 124 us at 1 slab (M=64), 238 at 3 slabs; qkvz (42 MB) M=96 52 us vs 55 at M=64.
    Exactness: M<=64, 128, 192, 256 bitwise equal to stock; other M max abs diff 4-8 on |y|~700-1500 (1 bf16 ulp; fp32 reduction order). NaN-sentinel guard: no writes past M for M=1..256, 3 shapes, all modes.
  conclusion: Interpretation (leader-6): shapes that fit in L2 now cost ~1 slab; gate_up/down (> 72 MB L2) still pay ~one DRAM stream per slab (2 x ~93 us mem roof ~ 188 us): parallel slabs are not time-aligned on the same weight stripe. Next: slab-inner stripe ordering (each CTA runs all slabs on a stripe back-to-back so the 2nd read hits L2).
validity:     exact up to fp32 reduction order (1 ulp max), masked, no OOB writes (guard test). Not used in serving yet.
  warm (job 35, mb4.py; first min(40MB, half) of each weight prefetched into L2): stock/new ms M=48 12.32/12.31, 64 14.67/14.65,
    72 21.97/20.30, 96 24.54/20.44, 120 31.01/23.99, 128 24.03/24.04, 144 32.13/32.69, 176 36.88/34.50. Warm 48->64 step +2.35 ms (in-server +3.0, T5-003).
