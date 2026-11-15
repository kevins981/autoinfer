team:         T3
builds_on:    T2-009 + T3-005 (T3-005's model.py delta ported onto the champion T2-009)
performance:  official #1: 919.0 tok/s (+1.33% vs T2-009 906.9; GSM8K 92.8%, all gates pass). Unofficial: +1.86% ABBA job 110 (all B > all A)
summary:
  hypothesis: T3-005's fix (keep the prefetch hint for the next layer's in_proj into the deferred MLP all-reduce)
              is independent of T2-009's fused norm/act+quant, so it should add the same ~1.5-1.9% on T2-009.
  change:     ws t3-c9 = T2-009 + T3-005's model.py delta (patch from T3-003 -> T3-005 code, applied cleanly):
              no hint clear after the MLP; next-layer lookup via the _PF_LAYERS prefix registry. Checked that in
              T2-009 both all-reduce branches of forward_fused still call tensor_model_parallel_all_reduce (the hostmem
              AR that consumes the hint).
  results:    Observed: same-job A/B, 200 prompts, T2-009 serve args, A = ws t3-c9A (T2-009), B = ws t3-c9:
              job 110: all B > all A, +1.86% (B 878.3/872.7/871.1 vs A 859.0/858.1/857.3); acceptance 2.335-2.350.
              The same delta on T3-003 (T3-005 vs T3-003): job 108 +1.86% all B > all A; job 101 +0.85% mixed.
  conclusion: Interpretation: the in_proj prefetch adds ~1.5-1.9%, independent of T2-009's fusion.
validity:     (leader-3) Prefetch-only change in model.py (where the hint points); the hint is always a static weight or
              None, so results are bitwise unchanged by construction. No memory change. Reviewed the delta vs T2-009:
              20 changed lines, model.py only.

official #1 (2026-11-15T00:09): 919.0 tok/s at c=16 (mean of 2 trials: 922.8 / 915.2, CV 0.6%) (+44.5% vs BASE), GSM8K 92.8%, MTP acceptance 2.331, TTFT p50 157 ms, TPOT p50 15.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
