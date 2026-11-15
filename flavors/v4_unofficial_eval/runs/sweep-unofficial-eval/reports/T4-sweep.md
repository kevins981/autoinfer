# T4 sweep (leader-4) — summary
Files: [1-inputs](T4-sweep/1-inputs.md), [2-graph](T4-sweep/2-graph.md), [3-calibration](T4-sweep/3-calibration.md)
(job 199 nsys of HEAD T3-004), [4-knobs](T4-sweep/4-knobs.md), [5-compare](T4-sweep/5-compare.md), [6-ranking](T4-sweep/6-ranking.md).
- Calibration: communication = 25% of GPU time (NCCL in mixed steps 12.5%, host AR in decode 12.5%); W8A16 decode GEMMs at
  0.79 TB/s; FP8 W8A8 reaches ~180+ TF in big mixed steps (so the FP8/fp32-acc peak is 330 TF, not 165).
- My top phase-1 candidate (duplex/pipelined host AR) was already measured negative by T1/T3; dropped before GPU spend.
- fp16-accumulate W8A8 (R4): asked; ruled NO (final, human).
- Current #1: layer-level two-micro-batch overlap of prefill all-reduces in mixed steps (P4), gated by a 2-GPU microbench.
