team:         T6
builds_on:    T6-002
performance:  "microbench only (cold, GPU0 per-step Marlin): M=72 25.0->18.8, 96 27.6->19.0, 112 29.5->22.7, 144 33.2->26.7 ms; worse at 160-192 (+1.3) and on down/gdn_out at 72-96 => per-shape policy needed. M<=64 unchanged (2x32 hand-mix est. 15.9 vs 17.1)."
summary:
  hypothesis: parallel slabs re-fetch >L2 weights because they are not time-aligned; slab-inner tile order (col = t/parallel, slab = t%parallel) makes the 2nd slab hit L2.
  change:     ws t6-slab2: marlin.cu + marlin_template.h, env AI_SLAB_INNER=1 (default off), flag bit 30 of prob_m. Diff: slab2_marlin.diff.
  results:    Observed (jobs 40, 41, 43; leader-6.w1): per-step cold ms stock/AI_SLAB/inner: 48 15.45/15.45/15.45; 64 17.07/17.07/17.04; 72 25.01/21.80/18.79; 80 25.23/21.92/18.95; 96 27.59/22.09/19.02; 112 29.45/25.08/22.71; 120 33.65/25.02/22.78; 128 25.10/25.12/22.77; 144 33.18/34.30/26.72; 160 35.46/34.33/35.67; 176 38.13/34.41/35.70; 192 34.54/34.53/35.80. Warm similar (e.g. 96: 24.92 -> 18.77).
    gate_up us M=96 stock/AI_SLAB/inner 215/191/125; down 96/79/90. ncu gate_up M=96: DRAM 134.8 MB (AI_SLAB) -> 92.0 MB (inner, = 1 weight stream); ncu clocks show a smaller time gain (186 -> 158 us) than graph/event timing (191 -> 125), unresolved.
    Exactness: 816 comparisons (3 shapes, M=1..256, incl. 2x32): 194 bitwise, max abs diff 4.0 (1 bf16 ulp at |y|~700), no non-finite, 0 guard violations.
  conclusion: Interpretation: the extra traffic was misaligned slabs; slab-inner fixes it for gate_up (~1-slab cost at M=72-96). Best-of per shape est. M=96 ~17.9 ms = 1.02x roof. Kernel remains latency-bound at 1 CTA/SM (ncu: issue-active 12-17%, tensor 34-43%).
validity:     exact up to fp32 reduction order (<=1 ulp); masked; guard test clean. Not serving yet.
