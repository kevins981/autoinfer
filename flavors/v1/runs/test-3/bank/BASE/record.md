team:         manager
performance:  (official results appended below by official_eval)
summary:
  hypothesis: none: the task baseline
  change:     none (empty change.diff); serve args = task/harness/serve_args.sh
  results:    see official/
  conclusion: reference for this run's latency gates
validity:     the task baseline by definition

official #1 (2026-11-06T22:53): 631.4 tok/s at c=16, GSM8K 92.8%, MTP acceptance 2.34, TTFT p50 251 ms, TPOT p50 23.2 ms; gates: GSM8K pass, latency n/a (official/1/results.json)

official #2 (2026-11-07T03:48): 635.9 tok/s at c=16 (mean of 2 trials: 635.9 / 635.9, CV 0.0%), GSM8K 94.4%, MTP acceptance 2.345, TTFT p50 235 ms, TPOT p50 23.0 ms; gates: GSM8K pass, latency n/a, 128K request pass (official/2/results.json)
