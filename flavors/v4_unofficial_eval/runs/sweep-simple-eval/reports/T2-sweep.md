# T2 sweep (leader-2): summary
Map files: [1-inputs](T2-sweep/1-inputs.md), [2-graph](T2-sweep/2-graph.md), [3-calibration](T2-sweep/3-calibration.md),
[4-knobs](T2-sweep/4-knobs.md), [5-compare](T2-sweep/5-compare.md), [6-ranking](T2-sweep/6-ranking.md).
- Calibration: BASE benchmark = decode steps 97 s (39.9 ms x 2,430) + mixed-prefill-step fixed excess
  19 s (48 ms each) + prompt rows 25 s (0.244 ms/row): 141.7 s modelled vs 143.4 s measured.
- Framework gap found: no Host (CPU launch / cross-rank skew) resource; it explains the mixed steps'
  fixed 48 ms (eager above 96 tokens, NCCL waits on the slower rank).
- Top candidates: W1 prefill CUDA graphs to 2048 tokens (~+13%), faster AR (T1), Marlin efficiency,
  reduced-vocab + FP8 draft lm_head, W8A8 for mixed steps, MTP k=3 after cheaper drafts.
