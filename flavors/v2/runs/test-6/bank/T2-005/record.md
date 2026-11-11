team:         T2
builds_on:    T1-042 (clean, no change)
performance:  "INVALID (job 19 ran with the prebuilt baseline _C lib: GDN replay and T1-042 csrc changes absent; superseded by T2-006). measurement only: critical-path anatomy of the champion's mixed steps (fills the gap in KB56)"
summary:
  hypothesis: Mixed steps (~39% of wall) have non-GEMM levers worth a direction.
  change:     none. nsys TRACE=node, 64 prompts c=16, champion serve_args (job 19, by leader-2.w1). Exclusive attribution on rank 0 along the union timeline of all kernels+memcpys in each step's GPU span, priority marlin > hostar > GDN > attention > norm/act > lm_head > memcpy (side-stream overlap is not double counted, KB58). Script agents/leader-2.w1/anat/cp.py; tables anatomy_tables.txt.
  results:    Observed (median ms per step, rank 0; nsys inflates host side ~3%):
    | family | decode | mixed pt<150 (med 66 prefill tok) | pt 150-399 (med 314) | pt>=400 (med 731) |
    | period | 26.3 | 79.8 | 95.2 | 140.5 |
    | Marlin | 15.2 | 35.1 | 84.0 | 157.7 (two streams, clipped) |
    | host-AR kernel wait | 6.1 | 10.2 | 10.3 | 30.7 |
    | AR memcpy not hidden | 0 | 0.03 | 6.5 | 13.0 |
    | GDN | 3.2 | 6.1 | 8.2 | 11.0 |
    | norm/act/elementwise | 0.4 | 3.2 | 3.7 | 5.6 |
    | attention (+fused add-rmsnorm) | 0.8 | 1.5 | 2.2 | 3.4 |
    | lm_head/small GEMMs | 1.0 | 1.2 | 1.3 | 1.5 |
    | GPU idle in span | 0.1 | 1.6 | 4.7 | 4.5 |
    | outside GPU span (host) | ~0 | 17.1 | ~0 | ~0 |
    Step counts in this run: 494 decode, 22 / 8 / 19 mixed (x6.25 for 400 prompts; ramp-up inflates large-pt steps).
    GDN in mixed steps: fused_sigmoid_gating_delta_rule_update ~4.0 ms/step (the decode tokens' recurrent update), chunked prefill kernels ~4 ms. Elementwise: act_and_mul and add_kernel dominate.
    Marlin at pt~314: 84 ms vs 71 ms for 8.7 TFLOP at 123 TF (~85% of the microbench rate).
  conclusion: Interpretation: in mixed steps, Marlin is near its roof. The non-GEMM levers are AR exposure (17-44 ms/step), a ~17 ms host-side gap in small-prefill mixed steps, unfused elementwise ops (3-6 ms), and the mixed-step GDN decode update (4.0 vs 2.3 ms in decode steps). Each is a few % of wall.
validity:     measurement only.
