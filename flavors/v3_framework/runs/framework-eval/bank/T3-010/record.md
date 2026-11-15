team:         T3
builds_on:    T3-009
performance:  unofficial: -1.15% (ABBA job 128, all B < all A)
summary:
  hypothesis: 31 of 64 mixed steps (job 117) have < 256 rows and so no row-split; lowering VLLM_ROWSPLIT_MIN_TOKENS to
              128 should hide part of their AR (~0.5-1% napkin).
  change:     env only: B = VLLM_ROWSPLIT_MIN_TOKENS=128 on T3-009 code (same workspace t3-rs11 both arms).
  results:    Observed: A 911.1/899.0 vs B 897.5/891.8 tok/s, -1.15%, all B < all A; acceptance 2.321-2.341.
  conclusion: Interpretation: halves of 64-127 rows (<= 1.31 MB) go to the SM-spinning hostmem AR on the side stream,
              whose 32 CTAs slow the concurrent GEMMs badly (T3-002: hm||gemm +87-333 us). Revisit only if the CE AR
              (no SM spinning) wins on the side stream (worker task 6).
validity:     env-only, exact.
