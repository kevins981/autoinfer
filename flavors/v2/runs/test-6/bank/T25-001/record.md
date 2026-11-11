team: T25
builds_on: T14-004
performance: "measurement only (microbench, no end-to-end change)"
summary:
  hypothesis: GPU1 (SM 2520-2685 MHz) can take ~6-8% more MLP intermediate columns than GPU0 (locked 2235 MHz) in compute-bound steps, equalizing per-rank MLP time.
  change: none (microbench only: Marlin FP8 gate_up/down with N/K scaled by f in .88..1.12, M=48..640, both GPUs; agents/leader-25.w1/G1/)
  results: Observed: at f=1 GPU0 is +17/+11/+17/+19% slower than GPU1 at M=128/256/384/640 (cold, gate_up+down), +3.9% at M=48. Line-fit balance f0/f1 = .98/1.02 (M48), .96/1.04 (128), .94/1.06 (256), .94/1.06 (384), .92/1.08 (640); MLP saving per layer 2/10/46/38/101 us (1.2-8.9%). Time vs f is bumpy (Marlin config switches). Offset-view weight identical in time and bitwise output.
  conclusion: Interpretation: rebalance worthwhile only for compute-bound mixed steps (M>=256), ~5-9% of MLP time; negligible for decode. Details in REPORT.md.
validity: no code change; nothing to review.
