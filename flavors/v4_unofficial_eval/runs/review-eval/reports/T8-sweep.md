# T8 sweep (leader-8): summary
Map: [1-inputs](T8-sweep/1-inputs.md), [2-graph](T8-sweep/2-graph.md), [3-calibration](T8-sweep/3-calibration.md),
[4-knobs](T8-sweep/4-knobs.md), [5-compare](T8-sweep/5-compare.md), [6-ranking](T8-sweep/6-ranking.md).
The team opened ~07:04 UTC, 71 min before the run's end, so the map is compact. Phase 1's cheap candidates (k=3, larger L2 prefetch)
had already been tested by T7 and T5. The new dimension found, fp16 accumulation inside a K-block, was measured (job 124: 1.09-1.23x on
prefill GEMMs) and then ruled not allowed (bank/T8-001). Main lesson: fp32-acc W8A8 runs at 240-275 TFLOP/s, so the mixed-step cost is
mostly non-GEMM overhead. That is the top remaining lever (6-ranking #1), but it is too large to build before the run ends.
