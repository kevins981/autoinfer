team:         T1
builds_on:    T2-004
performance:  unofficial: PASS +5.07% vs T2-004 (731.3 / 725.9 vs 693.4), 128K pass
summary:
  hypothesis: rebase of T1-002 (host-pinned one-shot all-reduce for <=128-row ARs, PASS +3.40% vs T1-004) onto HEAD T2-004 (prefill CUDA graphs up to 512 tokens). Decode ARs are unaffected by T2-004, so expect ~+3% again; mixed steps <=128 tokens now replay graphs that include the host AR (graph-capturable by design).
  change:     T1-002's change.diff (unchanged) + T2-004's serve args.
  results:    Observed: unofficial #1 PASS +5.07% vs HEAD T2-004 (B 731.3/725.9 vs ref 693.4); 128K pass; p99 TTFT 1130 / TPOT 44.7 ms; acceptance 2.354.
  conclusion: Interpretation: larger than T1-002's +3.40% vs T1-004. Plausible reason (unverified): with T2-004 the mixed steps up to 512 tokens replay graphs, so their AR time is no longer hidden behind CPU dispatch, and mixed steps <=128 rows now also use the faster AR. Cross-job reference noise (~1%) also applies.
validity:     As T1-002 (bitwise-equal all-reduce, fixed grid, startup self-test); serve args are HEAD's. Prefill graphs now capture the host AR at more sizes: exercised by the self-test + stress (CUDA-graph mixed sizes) already; --long run.

unofficial #1 (2026-11-15T06:19): PASS: +5.07% vs HEAD T2-004 (B 731.3 / 725.9 vs reference mean 693.4 over 2 starts, 0 run in this job; 200 prompts); 128K pass, p99 TTFT 1130 ms / TPOT 44.7 ms, MTP acceptance 2.354 (unofficial/1/unofficial.json)

official #1 (2026-11-15T06:32): 760.8 tok/s at c=16 (mean of 2 trials: 760.1 / 761.4, CV 0.1%) (+19.6% vs BASE), GSM8K 93.6%, MTP acceptance 2.341, TTFT p50 194 ms, TPOT p50 19.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
