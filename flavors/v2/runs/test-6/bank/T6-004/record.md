team:         T6
builds_on:    T6-003 (+ T1-042)
performance:  "microbench (GPU0 per-step Marlin, cold|warm): M<=48 unchanged (bitwise); 52-64 -3%; 72 -29%|-18%; 80 -30%|-19%; 96 -35%|-25%; 112 -21%|-13%; 120 -31%|-23%; 144 -14%|-9%; 176 -10%|-5%. In-server: pending."
summary:
  hypothesis: a per-shape/per-M choice among stock / single-launch parallel slabs / slab-inner / 2x32 gives near-roof Marlin for M=65-191 without touching M<=48.
  change:     ws t6-pol: marlin.cu t6_knobs() policy (AI_POLICY unset = on, =0 stock, =manual knobs) + marlin_template.h per-slab masking/ordering. Table: POLICY.md. Also min_workspace_size 2*sms (production sms*4).
  results:    Observed (job 51, full grid, same build): see performance line; M=64 16.61 vs 17.13 cold; M=96 17.97 (1.02x roof). Exactness M=1..256 x 6 shapes: max abs 4.0 (1 bf16 ulp at |y|~700), 0 guard violations, 0 non-finite; bitwise at M<=48 and several ranges.
    Occupancy probes (job 46, t6-opt): 2 CTAs/SM and 256-thread tmb2-4 tiles: no gain at M<=48 (best-per-shape -0.9% cold / -1.9% warm at M=48). Stage-3 thread_k=64 kernels are INCORRECT (group-scale prefetch gated on pipe%2): mb9 gains for c128_s3/bps2_s3 are artifacts, not used.
  conclusion: Interpretation: G1 met for 96-112, borderline 80, missed at 72 (2x48 padding) and 64 (16.61 vs 16.0). The occupancy knobs reachable cheaply don't fix the M<=64 latency-bound kernel.
validity:     exact up to fp32 reduction order; masked; KB61 check in job log (_C from t6-pol). Leader review: reorder only, no precision change; M<=48 decode path bitwise unchanged.
