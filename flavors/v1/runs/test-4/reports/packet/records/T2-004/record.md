team:         T2
performance:  "unofficial: 630.9/636.0 tok/s (mean 633.5) vs T2-003 634.4 and BASE 611.4, same job, 200 prompts: equal to T2-003"
summary:
  hypothesis: greedy target verification from a vocab-parallel argmax (exchange (value,index) pairs, not 48x248K logits)
              saves ~1 ms/decode step (+2-3%).
  change:     builds on T2-001 (same serve_args). New spec_decode/greedy_fastpath.py (vocab_parallel_argmax, Triton greedy_verify);
              RejectionSampler.can_use_greedy_fastpath/verify_greedy_from_argmax; ModelRunner._try_greedy_fastpath hook in
              sample() (non-batch-sharded branch only). Generic condition: all requests temperature 0, no logits processing /
              masking / logprobs / grammar / draft logits / special rejection methods / adaptive verification / watermark /
              nan count / sampling mask / trace replay; logit scale 1, no soft cap; else the unchanged full path.
  results:    Observed: numeric test 0/9600 mismatches vs torch.argmax over the full row (ties across/within shards, -inf rows,
              padded vocab; 2-shard reduction emulated in one process); greedy_verify == rejection_sample(T=0) on 300 random
              batches. Server: fast-path log line on both ranks; acceptance 2.344/2.349; gsm8k-50 88.0% (=BASE);
              TPOT p99 42.3/56.8, TTFT p99 1311/1393. Env var used for the A/B was removed afterwards (syntax-checked only).
  conclusion: Interpretation: exact and working, but no measurable gain over the existing --enable-batch-sharded-sampling
              (T2-003), which already captures most of the target-logits exchange cost. Not sent to official eval.
validity:     leader-2: read the diff. Exact argmax (lowest index on ties, matching baseline tl.argmax left tie-break),
              verification logic equals the greedy rejection path; condition keyed only to request params. Known divergence:
              baseline maps NaN logits to -inf before argmax, fast path doesn't (only with NaN logits). OK for correctness.
