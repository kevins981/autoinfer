# T4 sweep summary (leader-4)
Map files: [1-inputs](T4-sweep/1-inputs.md), [2-graph](T4-sweep/2-graph.md), [3-calibration](T4-sweep/3-calibration.md),
[4-knobs](T4-sweep/4-knobs.md), [5-compare](T4-sweep/5-compare.md), [6-ranking](T4-sweep/6-ranking.md).
- Calibrated on the champion T3-009 (jobs 139/140) plus the benchmark's 400 real prompt lengths: decode-only steps ~55 s,
  mixed (prefill-carrying) steps ~42 s of the 96 s wall (napkin 96.9 s vs measured 95.9 s).
- Decode candidates on my map are all built or measured dead by T1-T3.
- What's left is in mixed steps:
  1. CUDA graphs past 512 tokens (76 prompts run fully eager; earlier verdict used Marlin-era costs): +5-6%. Testing (E1).
  2. Small mixed steps CPU-bound in the eager GDN core: +9-11% if real, <1% if T2's gap measurement holds. Ground truth via a non-nsys step log in E1.
