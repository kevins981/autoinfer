team:         T30
builds_on:    T29-006
performance:  "measurement only (not a candidate): AR-off floor of mixed steps"
summary:
  hypothesis: in big mixed steps (pt>=209) a large part of the TP all-reduce is exposed, despite the token-split chain.
  change:     T29's untraced per-step probe (runner hooks only) + T30_NOAR=1 (all TP all-reduces become no-ops; timing only,
              outputs garbage). Both default off. By leader-30.w1; files agents/leader-30.w1/ar_floor/.
  results:    Observed: job 235 arm A (champion code, 200 prompts): 1007.3 tok/s, big-step fit 0.160 ms/token + 39.5 ms (n=83).
              Job 244 arm B (T30_NOAR=1, --ignore-eos, 60 prompts): fit 0.109 + 23.1 (n=24). Median GPU ms/step A->B:
              decode 24.9->21.8, pt1-80 43.7->35.2, pt81-208 49.4->35.9, pt209-400 93.6->62.1, pt401-1000 138.6->95.7, >1000 293->177.
              Job 235's B arm hung (garbage hit EOS on every request; the bench retried to the 15-min cap).
  conclusion: Interpretation: about a third of big-step GPU time is exposed AR (~3.5 s per 47.4 s of GPU at 200 prompts). The
              token-split chain joins at each core, so the ARs have little compute beside them. This motivates proposals/T30-1.md.
              Caveats: B has n=24 big steps, and garbage activations may change GDN/attention timing.
validity:     not for official eval (T30_NOAR breaks outputs by design; the flag is default off).
