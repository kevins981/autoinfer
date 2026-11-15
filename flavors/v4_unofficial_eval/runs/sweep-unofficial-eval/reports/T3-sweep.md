# T3 sweep (leader-3) — summary
Map files: [1-inputs](T3-sweep/1-inputs.md), [2-graph](T3-sweep/2-graph.md), [3-calibration](T3-sweep/3-calibration.md),
[4-knobs](T3-sweep/4-knobs.md), [5-compare](T3-sweep/5-compare.md), [6-ranking](T3-sweep/6-ranking.md).
- Calibration (HEAD T1-016, jobs 153/155): decode step 28.8 ms (napkin with measured efficiencies 27.7); mixed steps are
  ~37% of wall and cost a **fixed ~65 ms over a decode step** plus ~0.1 ms/token.
- New dimension vs T1/T2 maps: **prefill admission batching** (Window/Merge along r over time; a Policy-layer choice
  the framework leaves out). vLLM's generic `--prefill-schedule-interval N` implements it. Napkin ~8%.
- Win: T3-004 target local argmax (+2.11%, HEAD). Updates: update-1..3; write-up ../T3-writeup.md.
- Most of my other candidates were already tested by T1/T2 (AR speed, L2 prefetch, 3 drafts, tree, lazy K=3) or built
  (T1-017 W8A8 prefill route).
