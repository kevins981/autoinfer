team:         T3
builds_on:    T2-011 + T3-008 (T3-008's model.py row-split delta ported onto the champion T2-011)
performance:  official #1: 951.5 tok/s (+2.11% vs T2-011 931.8; GSM8K 93.2%, all gates pass). Unofficial: +1.40% ABBA job 123 (all B > all A)
summary:
  hypothesis: As T3-008: restore the row-split AR overlap in mixed steps that T2-009's fused norm+quant switched off.
  change:     ws t3-rs11 = T2-011 + T3-008's model.py delta (patch from T3-006 -> T3-008 code, applied cleanly; T2-011
              changes only lm_head.py / logits_processor.py).
  results:    Observed: ABBA job 123 (A = ws t3-c11 = T2-011, B = t3-rs11, 200 prompts, T2-011 args): B 903.4/897.9 vs
              A 889.1/887.2 tok/s, +1.40%, all B > all A; acceptance 2.341-2.349. Row-split log line in both B servers, in
              neither A server. Same delta on T3-006: +1.58% all B > all A (job 121, T3-008).
  conclusion: Interpretation: the overlap restore is independent of T2-011's draft head; ~1.4-1.6%.
validity:     (leader-3) As T3-008: per-row-identical computation, exact by construction; mixed steps use the same fused
              norm+quant numerics as the champion's decode steps. Delta vs T2-011: model.py only. No memory change.

official #1 (2026-11-15T01:06): 951.5 tok/s at c=16 (mean of 2 trials: 955.0 / 947.9, CV 0.5%) (+49.6% vs BASE), GSM8K 93.2%, MTP acceptance 2.33, TTFT p50 151 ms, TPOT p50 15.5 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
