team:         T2
performance:  "OFFICIAL #1: 653.9 tok/s (652.1/655.6), +2.8% vs BASE; all gates pass"
summary:
  hypothesis: the existing batch-sharded sampling (target logits all-to-all, each rank samples half the batch) halves the
              target logits exchange (~0.96 ms/step) and the sampling work.
  change:     T2-001 flags + --enable-batch-sharded-sampling. Empty diff. Builds on T2-001.
  results:    Observed (same job, 150 prompts): 616.5 tok/s, TPOT med/p99 22.7/50.5, TTFT p99 1191, acceptance 2.355 (.786/.570).
              Log: "Batch-sharded sampling enabled".
              Job 2 (200 prompts, two benches on one server): 634.8 / 633.9 tok/s, TPOT p99 51.9/52.2, TTFT p99 1188/1491,
              acceptance 2.343/2.344; BASE in same job (1 bench) 611.4, TPOT p99 51.2, TTFT p99 1362, acc 2.356.
  conclusion: Interpretation: ~+3.5-4% over BASE, reproducible across two jobs; equal to the custom greedy fast path
              (T2-004, 633.5 mean) so the flag-only version is preferred. Sent to official eval.
validity:     leader-2: flags only; batch-sharded sampling computes the same full-vocab logits per request (all-to-all of vocab
              shards) and the same rejection sampler; exact. OK.

official #1 (2026-11-07T04:52): 653.9 tok/s at c=16 (mean of 2 trials: 652.1 / 655.6, CV 0.4%) (+2.8% vs BASE), GSM8K 93.2%, MTP acceptance 2.348, TTFT p50 228 ms, TPOT p50 22.5 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
