team:         T1
builds_on:    T1-001 + T1-002
performance:  "official #1: 707.6 tok/s (+11.3% vs BASE), all gates pass"
summary:
  hypothesis: T1-001 (cheaper drafter: fp8 draft lm_head + local draft argmax) and T1-002 (host-memory all-reduce for
              decode-size all-reduces) cut different parts of the decode step (draft lm_head/gathers vs body ARs), so the
              gains should roughly add (~+7%).
  change:     code of T1-002 (HOSTMEM all-reduce backend, with the acquire fence) + serve flags of T1-001.
  results:    Observed: official #1 707.6 tok/s (709.6/705.6), GSM8K 94.4%, acc 2.344, TTFT p99 1213 ms, TPOT p99 42.9 ms, 128K pass.
  conclusion: Interpretation: +11.3% vs ~+7% expected from the separate unofficial gains (BASE official sits low in its noise band; also the AR saving may compound with the faster drafter: drafter ARs are decode-size too).
validity:     union of T1-001 and T1-002's reviews: exact (draft side fp8 head per ruling 1, local argmax exact for greedy
              drafts; all-reduce bitwise equal to NCCL, bf16 comms). Memory: fp8 draft head copy +0.64 GB/GPU and ~5 MB pinned
              host; the 128K request passed with a config using more memory (T1-003: same flags, k=3).

official #1 (2026-11-14T17:14): 707.6 tok/s at c=16 (mean of 2 trials: 709.6 / 705.6, CV 0.4%) (+11.3% vs BASE), GSM8K 94.4%, MTP acceptance 2.344, TTFT p50 218 ms, TPOT p50 20.9 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
