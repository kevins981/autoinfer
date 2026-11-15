# T5 sweep (leader-5) — summary
Files: [1-inputs](T5-sweep/1-inputs.md), [2-graph](T5-sweep/2-graph.md), [3-calibration](T5-sweep/3-calibration.md)
(jobs 201 nsys of HEAD T3-004, 202 NVTX-only 200 prompts), [4-knobs](T5-sweep/4-knobs.md), [5-compare](T5-sweep/5-compare.md),
[6-ranking](T5-sweep/6-ranking.md).
- Calibration: decode step 27.6 ms = W8A16 GEMMs 15.9 (0.79 TB/s) + host AR 5.7 (135 x 42 us) + lm_head 1.6 + GDN 1.8 + rest.
  My mixed-step CPU attribution was wrong (corrected in 5-compare from T2-014 / T3 job 198 / T4 job 199).
- Most of my phase-1 list was already tested negative (pipelined AR, in_proj_ba, k=3).
- New dimension: **lossless encoding of the all-reduce payload** (bf16 exponent bytes are predictable): bit-exact, ~23%
  fewer link bytes, ~4% napkin (decode AR + prefill AR). Ruling requested; offline compressibility check first.
