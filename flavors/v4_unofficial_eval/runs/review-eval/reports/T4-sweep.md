# T4 sweep (leader-4): summary
Map: [1-inputs](T4-sweep/1-inputs.md), [2-graph](T4-sweep/2-graph.md), [3-calibration](T4-sweep/3-calibration.md),
[4-knobs](T4-sweep/4-knobs.md), [5-compare](T4-sweep/5-compare.md), [6-ranking](T4-sweep/6-ranking.md) (with addendum and
review answers). Review: reviews/T4-1. Measurements: GPU job 66 (nsys of T3-001), agents/leader-4.w1/analysis.md, mixed.md.

Main finding (not in the other maps): mixed steps (a prompt + ~15 decoding requests) are CPU-launch-bound. A ~60-row mixed step
costs 72-122 ms vs 32.7 ms for a decode step; its GPU work is ~50 ms; ~1730 eager launches per mixed step (GDN prefill FLA chunk
pipeline + the general GDN core that decoding rows fall back to + ~1000 tiny copies), GPU1's CPU thread lags and GPU0 waits in
NCCL. Mixed steps are ~32% of step time in the profile.
Ranking: (1) cut mixed-step GDN launches: one-kernel recurrent prefill for short prompts (03 §4.2 dual form) and keep the fused
CUDA spec-decode kernel for decoding rows inside mixed batches; (2) GDN DeltaLog (1 state write instead of 3 per verify);
(3) k=3 drafts if GDN state capacity allows; (4) smaller token budget (conditional on 1).
