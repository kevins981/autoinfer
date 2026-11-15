team:         T1
builds_on:    T1-001
performance:  unofficial: PASS +1.61% vs T1-001 (656.3 / 671.0 vs 653.1)
summary:
  hypothesis: the target's full-logits all-gather costs ~0.96 ms per decode step (48 rows x 124160 x bf16 per GPU, BASE profile). Batch-sharded sampling (existing option; Qwen3.5 implements compute_logits_local) has each rank sample half the requests after an all-to-all of logit shards: about half the link bytes, and half the sampling work per rank. Napkin: ~0.5 ms per decode step, ~1-1.5%.
  change:     serve args: T1-001's + --enable-batch-sharded-sampling
  results:    Observed: unofficial #1 PASS, B 656.3 / 671.0 vs HEAD ref mean 653.1 (+1.61%; first start only +0.5%); MTP acceptance 2.349 (HEAD 2.353); p99 TTFT 1106 / TPOT 40.0 ms. Server log: 'Batch-sharded sampling enabled.' on both ranks.
  conclusion: Interpretation: consistent with the napkin (~0.5 ms/decode step of logits traffic + half the sampling per rank, ~1-1.5%); the starts spread widely (656-671), so the size is uncertain; acceptance unchanged as expected for an exact change.
validity:     Same sampler math on the same full-vocab logits, only redistributed by request instead of gathered on every rank; greedy verification unchanged, so outputs are exact. Logprobs stay supported (fixed-width gather). Memory: negligible. Nothing benchmark-specific.

unofficial #1 (2026-11-15T05:33): PASS: +1.61% vs HEAD T1-001 (B 656.3 / 671.0 vs reference mean 653.1 over 2 starts, 0 run in this job; 200 prompts); p99 TTFT 1106 ms / TPOT 40.0 ms, MTP acceptance 2.349 (unofficial/1/unofficial.json)
