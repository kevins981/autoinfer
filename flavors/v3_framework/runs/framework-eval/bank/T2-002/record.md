team:         T2 (combination, credit T1 + T2)
builds_on:    T1-005 + T2-001
performance:  "official #1: 757.6 tok/s (+19.1% vs BASE, +7.1% vs T1-004 official 707.6), all gates pass"
summary:
  hypothesis: T1-005's batch-sharded sampling (+1.78% unofficial on T1-004) and T2-001's mixed-step CUDA graphs
              (+6.42% unofficial on T1-004) act on different parts (target logits gather/sampling vs eager mixed steps),
              so they should roughly add (~+8% over T1-004).
  change:     T1-005's code (= T1-004's) + T1-005's serve flags + T2-001's cudagraph_capture_sizes.
  results:    Observed: official #1 757.6 tok/s (756.3/758.9), GSM8K 93.2%, acceptance 2.342, TTFT p50 194 ms, TPOT p50
              19.5 ms, 128K pass (official/1/results.json).
  conclusion: Interpretation: the two gains roughly add (+1.78% + 6.42% unofficial -> +7.1% official over T1-004).
validity:     Union of T1-005's and T2-001's reviews: both are engine options/flags; exact (sampling per request from
              full-vocab logits; graphs run the same kernels). Memory: batch-sharded sampling has no notable memory change
              (T1-005 record); T2-001 passed the 128K request with T1-004's flags; the official eval re-runs the 128K gate.
              Agreed with leader-1 (T1) to file as one combination record.

official #1 (2026-11-14T17:57): 757.6 tok/s at c=16 (mean of 2 trials: 756.3 / 758.9, CV 0.2%) (+19.1% vs BASE), GSM8K 93.2%, MTP acceptance 2.342, TTFT p50 194 ms, TPOT p50 19.5 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
