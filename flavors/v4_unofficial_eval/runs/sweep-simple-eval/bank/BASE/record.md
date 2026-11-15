team:         -
builds_on:    -
performance:  official results below
summary:
  hypothesis: none: the task baseline
  change:     none (empty change.diff); serve args = task/harness/serve_args.sh
  results:    see official/
  conclusion: the reference for every comparison and for the latency gates
validity:     the task baseline by definition

official #1 (shipped with the task, task/baseline/official/): 635.9 tok/s at c=16, GSM8K 94.4%, MTP acceptance 2.345, TTFT p50/p99 235/1341 ms, TPOT p50/p99 23.0/46.1 ms (official/1/results.json)

unofficial #1 (2026-11-15T15:27): 634.5 / 643.4 tok/s (200 prompts); p99 TTFT 1240 ms / TPOT 40.3 ms (BASE official 1341 / 46.1), MTP acceptance 2.357 (bank/BASE/unofficial/1/unofficial.json)
