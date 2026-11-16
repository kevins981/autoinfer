team:         T1
builds_on:    BASE
performance:  +4.2% vs BASE (635.9), -2.4% vs the champion T1-001 (678.4): 662.4 tok/s, starts 660.8 / 663.9 (eval/1, GPU job 7)
summary:
  hypothesis: the drafter only needs greedy argmax; a head over a generic token subset ids [0,K) (BPE merge
              order = the tokenizer's own corpus frequency, non-benchmark, ruling 2), replicated on both ranks,
              removes the draft logits gather and ~3/4 of the draft head bytes. Napkin +4.5% minus acceptance loss.
  change:     Python only (change.diff, by worker leader-1.w1): env VLLM_MTP_DRAFT_VOCAB=K (default off);
              load_eagle_model builds a replicated bf16 [K,5120] copy of target lm_head rows [0,K) (all-reduce of
              zero-padded shards, bit-exact rows); sample_draft does a local F.linear + greedy_argmax when drafting
              is greedy (falls back otherwise). Measured with K=32768 (env set in the job, not yet a code default).
  results:    Observed: 662.4 tok/s (660.8, 663.9); MTP acceptance 2.238 (per position 0.736/0.500,
              0.738/0.502) vs 2.345 BASE / 2.346 T1-001; TTFT p99 1240 ms, TPOT p99 46.2 ms (gate_latency pass);
              128K pass. Ids [0,32768) decode as ~30.4K Latin, 75 CJK, 2.3K other (worker's count).
  conclusion: Interpretation: the step got cheaper (throughput/acceptance: 1.042/0.954 -> ~8.5% shorter
              step, vs ~6.3% for T1-001's fp8 full head), but acceptance fell 4.6%: drafts often need ids >= 32K.
              Net it loses to T1-001. A larger K or a better generic subset might recover acceptance; the upside
              over T1-001 is at most ~2%, so parked. FP8 variant of this head written (agents/leader-1.w1/
              draft_vocab_fp8.diff), not tested.
validity:     Ruling 2 (reduced-vocab draft head from non-benchmark token stats; verification full-vocab, exact).
              Target lm_head/verification untouched. Reviewed by leader-1 (diff read). Not a champion candidate.

review (reviews/T1-1): measurements, construction, norm placement and fallbacks confirmed. Corrections:
- Token subset: describe as a tokenizer-prefix heuristic (ids [0,K) of the published tokenizer, chosen without
  benchmark data); "merge order = corpus frequency" is an unverified assumption.
- "At most ~2% upside" is a proxy, not a bound (restoring acceptance at unchanged step cost gives ~2.35%).
- Replay: the diff defaults the feature off; reproduce with `env VLLM_MTP_DRAFT_VOCAB=32768` before eval.sh
  (as in GPU job 7).
