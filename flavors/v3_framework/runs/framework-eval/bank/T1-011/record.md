team:         T1
builds_on:    T2-006
performance:  "unofficial: mixed, +0.26% (ABBA 200 prompts, B 789.0/781.4 vs A 783.9/782.6), worker job 49 — not significant"
summary:
  hypothesis: ~12-15 us of each ~41 us HOSTMEM all-reduce is fixed cost; fusing AR + residual add + (Gemma)RMSNorm into one
              kernel removes a launch and a pass: microbench 2.8 us/call at 42 rows -> ~0.4 ms/step (~1.2%).
  change:     hostmem_ar.cu fused kernel, hooked as the fallback of fused_allreduce_gemma_rms_norm (env VLLM_HOSTMEM_FUSED_NORM,
              default on); all-thread fence. LL-style data+flag packets evaluated and rejected (33% payload overhead > latency saved).
  results:    Observed: microbench (2-rank graph) unfused/fused 44.5/41.7 us at 42 rows, 48.7/46.4 at 48, 21.7/19.0 at 14.
              Residual output bitwise equal to unfused; normed output differs from FlashInfer gemma_fused_add_rmsnorm in <=6 of
              655K elements by 1 bf16 ulp (fp reordering). ABBA mixed +0.26%; acceptance unchanged; gsm8k 50 on B 90.0%.
  conclusion: Interpretation: a real ~0.3-1% gain is plausible but below what one 200-prompt ABBA resolves. Kept as a
              stackable candidate; not worth more GPU on its own.
validity:     exact up to 1-ulp fp reordering of the norm (allowed); bf16 comms unchanged; generic.
