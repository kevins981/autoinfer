team:         T1
builds_on:    T1-012
performance:  "unofficial: +1.0% same-job (A A2 715.3/715.0 vs B B2 726.0/718.8, 200 prompts c=16, ABBA fresh servers)"
summary:
  hypothesis: the FP8 draft lm_head is still ~0.6 GB/rank read per draft step; keeping only the first N token ids (frequent BPE
              merges) cuts it ~2.5x at ~no acceptance cost (run-1 change 8, +2.1% same-job there).
  change:     port of prior/baseline-run-1 patch 0008: SpeculativeConfig.draft_lm_head_vocab_size; vocab_subset_lm_head_copy
              (first N rows of the target head, re-sharded over TP via an exact sum-with-zeros all-reduce at load, then FP8);
              wired in share_target_lm_head; Eagle supported_options. Served with N=98304 (serve_args.txt). Worker leader-2.w8,
              ws leader-2.vocab.
  results:    Observed: N from coverage on non-benchmark text (vllm repo docs/*.md + vllm/*.py, 14.5M tokens, coverage.txt):
              65536 97.64%, 81920 99.13%, 98304 99.93%, plateau above. 49152 rows/rank, no padding.
              A/B (agents/leader-2.w8/out/ab1): A 715.3, B 726.0, B2 718.8, A2 715.0 tok/s -> +1.0%; accept 2.345 vs 2.343;
              TPOT p50 ~19.8 vs ~19.4 ms; KV 267,721 tokens both (kv memory fixed 5 GiB; ~0.38 GiB/rank freed as headroom).
              gsm8k50 88% (= baseline subset). Expected ~0..+0.5% (napkin), measured +1.0% +-0.7.
  conclusion: Interpretation: small real gain likely (both B trials above both A), size uncertain; stack with other changes.
validity:     leader-2 review: target head/verification untouched (exact); drafts are full-vocab ids (row i = token i, shard offset
              added in get_top_tokens); no padding rows (49152/rank) and _mask_outside_vocab handles padding anyway; N chosen from
              repo text, not benchmark data (ruling 2). Load-time all-reduce sums disjoint owners with zeros (exact copy). OK.
