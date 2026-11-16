# T1 sweep summary (leader-1)

Map files: [1-inputs](T1-sweep/1-inputs.md), [2-graph](T1-sweep/2-graph.md),
[3-calibration](T1-sweep/3-calibration.md), [4-knobs](T1-sweep/4-knobs.md),
[5-compare](T1-sweep/5-compare.md), [6-ranking](T1-sweep/6-ranking.md).

Headlines: communication is ~25% of BASE's time and runs at 22% of the PCIe link on mixed steps
(NCCL LL, 2 channels); mixed prefill steps are 32% of time with GEMMs at 46% of bf16 peak; the
draft side reads the bf16 lm_head twice per step (~7% of a decode step). Top candidates: faster
all-reduce, cheaper draft lm_head, a better GEMM for mixed steps, DBO on mixed steps, k=3 drafts.
