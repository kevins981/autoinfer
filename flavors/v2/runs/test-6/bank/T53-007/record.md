team:         T53
builds_on:    T53-006 (same harness)
performance:  "measurement only: two offset half-batches (2x M=32) run 1.31-1.47x the M=64 champion-like chain (V0b), even with ARs as interference-free spins; gate (<=0.92x) FAILS. Schedule closed."
summary:
  hypothesis: Splitting the M=64 decode batch into two M=32 half-batches on two streams, offset so that B's GEMM i
              follows A's GEMM i (B's weights partly L2-warm), hides each half's AR and small ops behind the other
              half's work. Optimistic bound: ARs are 35 us spins (16 CTAs, hostar's AR block count) with no link
              interference.
  change:     agents/leader-53.w1/ceil/bench_half.py + job_half.sh. Single GPU, 64 distinct layers (192 MB weights
              each), 2 CUDA graphs of 32 layers.
              - Half-batch A runs on the capture stream, B on a second stream in the same graph. B waits on A's
                event per GEMM ("gemm" offset) or per op ("op" offset).
              - Per half: the 13-op chain at M=32, with the fastest standalone-cold config per GEMM (optimistic):
                qkvz t31 290 (50.1 us), gdn_out t31 322 (23.5), gate_up bs 127 (98.3; t31 322 105.5),
                down t31 322 (53.2).
              - Small ops as the same spins; ARs = 35 us spins on 16 CTAs.
              - GDN dummy: half the state bytes per half, spin 18.5 us (optimistic, half of 37) or 37 us.
              - Reference in the same job: V0b (M=64, AR 39/50 spins + 8 in-AR prefetch blocks, champion configs).
              Job 464, 0.1 GPU-min.
  results:    Observed (us/layer, medians):
              - V0b 342.0 (repeat at the end: 341.9). V1b (AR spins, no prefetch) 387.4.
              - Half-batch, GDN 18.5: 448.7 per-GEMM offset (1.312x V0b); 449.9 per-op offset (1.316x).
              - Half-batch, GDN 37: 463.9 per-GEMM offset (1.357x); 502.1 per-op offset (1.468x).
              - Standalone cold at M=32 vs M=64: qkvz 50.1 vs 51.2, gdn_out 23.5 vs 24.5, gate_up 98.3 vs 99.4,
                down 53.2 vs 56.3.
  conclusion: Interpretation: the decode GEMMs are weight-streaming bound, so a half-batch GEMM costs ~95-100% of
              the full-batch one, and two half-batches pay the weights about twice. 2 x 225 = ~450 us of GEMM per
              layer vs ~230.
              - L2 reuse by the offset half is small: the weights are 16-89 MB against a 72 MB L2, and A is already
                streaming the next GEMM's weights.
              - Hiding the ARs (~89 us/layer) and small ops (~18) behind the other half can't pay for ~225 us of
                extra weight traffic.
              - The schedule is closed even as an upper bound.
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
