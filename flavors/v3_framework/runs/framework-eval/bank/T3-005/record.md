team:         T3
builds_on:    T3-003
performance:  official #1: 909.7 tok/s (+1.37% vs T3-003 897.4; GSM8K 93.6%, all gates pass). Unofficial ABBA mixed +0.85% (job 101)
summary:
  hypothesis: In T3-003 the prefetch of the next layer's in_proj (qkvz/qkv) never ran: the MLP's all-reduce is deferred
              into the next layer's input norm, and T3-003 cleared the hint right after self.mlp(). Keeping the hint
              should save ~10-15 us per qkvz/qkv GEMM (0.5-0.7 ms/decode step).
  change:     (worker leader-3.w1, ws t3-pf) model.py: no hint clear after the MLP; the next layer is found via a
              prefix registry (_PF_LAYERS) instead of an nn.Module attribute set in the model loop. hostmem_all_reduce
              .py: a debug log gated on VLLM_AR_PF_DEBUG (off).
  results:    Observed: nsys of T3-003 code (job 88, c=16, MB 0 vs 40): gate_up 113.2 -> 92.1 us after attention,
              132.4 -> 110.7 after GDN; qkvz/qkv unchanged (53.3/46.8); AR before gate_up +2.3 us. Debug jobs 89-98:
              the AR before in_proj had pf=None in all samples (T3-003); with the fix the sequence is gate_up, qkvz,
              gate_up, ... ABBA vs T3-003 code (job 101): mixed, +0.85% mean.
  conclusion: Interpretation: the gate_up prefetch (T3-003) gives ~1.3 ms/decode step in the profile; the in_proj
              prefetch is consistent with a small extra gain but the ABBA can't separate it from noise. Official eval
              requested as the decisive test (costs the same as a repeat ABBA).
validity:     (leader-3) Same as T3-003: prefetch only; the hint is always a static weight tensor or None, so a stale hint
              can only cost time, never change results. No memory change. Debug print off by default.

official #1 (2026-11-14T23:17): 909.7 tok/s at c=16 (mean of 2 trials: 907.4 / 912.1, CV 0.4%) (+43.1% vs BASE), GSM8K 93.6%, MTP acceptance 2.329, TTFT p50 160 ms, TPOT p50 16.3 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)

repeat same-job A/B (leader-3, job 108, order BAABBA, 200 prompts, A = ws t3-pfA = T3-003 code, B = ws t3-p5 = T3-005):
  ABBA: all B > all A: +1.86% (B 875.3/875.1/868.5 vs A 862.1/856.9/852.0); acceptance 2.337-2.354 both arms.
  With job 101 (+0.85%, mixed): 5 starts per arm, B mean 868.6 vs A 856.2 (+1.45%).
