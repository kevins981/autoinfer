team:         T1
performance:  "official #1: 730.9 tok/s (+15.8% vs BASE, +4.5% vs T1-005 699.2), GSM8K 93.6%, TTFT p99 1438, TPOT p99 41.3, accept 2.344, gates + 128K pass"
summary:
  hypothesis: at temperature 0 with nothing else reading the logits, target verification only needs the argmax; a vocab-parallel
              (max, argmax) gather replaces the ~0.95 ms full-logits all-gather per decode step.
  change:     T1-007 (code T1-005 + spec flags in serve_args.txt) + greedy fast path: LogitsProcessor.get_greedy_tokens (per-shard
              Triton (max, argmax) over valid columns, float32 [rows,2] all-gather, strict '>' cross-rank so rank 0 wins ties);
              greedy_argmax_with_value; RejectionSampler.verify_greedy_ids + greedy_ids_rejection_sample kernel (accept draft ==
              target argmax); Sampler.pure_greedy per request (temperature==0, no logits processing incl. penalties/bias/bad
              words/min_tokens stop masks/thinking budget, no logprobs/prompt logprobs); model_runner._try_greedy_ids_sample
              (only spec-decode batches, no grammar, no batch sharder, default compute_sample_logits, no watermark/adaptive/
              block/synthetic/trace/nans/sampling mask; else unchanged full path); compute_greedy_tokens on the Qwen3.5 classes.
              TEST-ONLY env VLLM_TEST_GREEDY_IDS_CHECK=1 (default off) runs both paths. Worker leader-1.w4, ws leader-1.greedyverify.
  results:    Observed:
              - unit test (torchrun x2, out/unit_test.py): shard 124160 bf16, rows 1..64, adversarial ties within/across ranks,
                +-inf, all-equal, shard boundary, last column; ids vs greedy_argmax on the full gathered row AND end-to-end
                accepted tokens/num_sampled vs the existing rejection_sample: 4764 checks, 0 mismatches.
              - in-server dual path on real logits (T1-007 args, out/s3C_fast_check/server.log): 600 steps, 25511 rows, 0 mismatches,
                467 rows with a tied max; both TP ranks same counts. Earlier run on T1-005 args: 600 steps, 0 mismatches.
              - MTP acceptance 2.341 vs 2.340 (T1-007). gsm8k50 88% (as all runs).
              - bench 400 c=16 same job: 732.6 vs 715.3 (T1-007); job 2 on T1-005 args 711.1 vs 700.6. Napkin ~+2.5%.
  conclusion: Interpretation: exact by construction and by check; small (~2%) but consistent-direction gain.
validity:     leader-1 review (per human ruling r2: math + numeric check of the changed op on the same inputs + GSM8K/acceptance):
              the fast path returns the same id as greedy_argmax over the concatenated, vocab-cut row: first index within a shard
              (same Triton partial kernel restricted to valid columns), float32 values exact for bf16, strict '>' across ranks so
              rank 0 (lower ids) wins ties = first index globally; padding/out-of-vocab excluded or fallback. Shown on 25511 real
              rows incl. 467 ties, 0 mismatches. Eligibility is per-request sampling params only (generic, not benchmark-keyed);
              anything that reads/modifies logits falls back. Note: "no custom logits processors" is checked as
              len(logits_processors)==3 (built-in count) - correct today but fragile. NaN rows: fast path treats NaN as -inf; full
              path behavior on NaN undefined (degenerate case). No persistent buffers -> not a memory change; T1-007's 128K test
              stands. OK.

official #1 (2026-11-07T02:01): 730.9 tok/s at c=16 (mean of 2 trials: 729.7 / 732.1, CV 0.2%) (+15.8% vs BASE), GSM8K 93.6%, MTP acceptance 2.344, TTFT p50 204 ms, TPOT p50 20.0 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
