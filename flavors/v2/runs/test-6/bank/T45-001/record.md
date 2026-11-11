team:         T45
builds_on:    T37-002
performance:  "CPU only, n/a (null: the current convention is the right one)"
summary:
  hypothesis: The MTP draft head might accept more if it were fed the target's PRE-final-norm hidden state rather than
              the post-norm one (qwen3_5_mtp.py feeds the post-norm hidden into pre_fc_norm_hidden).
  change:     None to the engine. Offline A/B on T3's held-out data (agents/leader-3.w1/data, post-norm hiddens,
              T3's offline MTP pipeline). The pre-norm direction is recovered exactly up to bf16 as H/(1+w_final),
              because RMSNorm is scale-invariant and pre_fc_norm_hidden renormalises. Script:
              agents/leader-45/mtpnorm/eval_norm.py.
  results:    Observed (n=14138 positions): post-norm (current) a1=0.7813, a2cum=0.5605; pre-norm a1=0.7614,
              a2cum=0.5354. The final-norm weight and pre_fc_norm_hidden weight correlate at -0.62.
  conclusion: Interpretation: the checkpoint's MTP head was trained on post-norm inputs; the current wiring is
              correct. Closes this question. The -0.62 correlation suggests pre_fc_norm_hidden partly undoes the
              final norm's per-channel gain.
validity:     No engine change; no GPU used.
