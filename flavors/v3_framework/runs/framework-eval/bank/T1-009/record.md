team:         T1
builds_on:    T2-002 + T1-008
performance:  "official #1: 786.4 tok/s (+3.8% vs champion T2-002 757.6, +23.7% vs BASE), all gates pass"
summary:
  hypothesis: T1-008's gain (+8.4% on T1-005, mostly decode-step CPU/sync idle) should largely survive on the champion T2-002
              (whose gain came from graph-capturing mixed steps).
  change:     champion T2-002 (code + serve args) + --attention-backend TRITON_ATTN.
  results:    Observed: official #1 786.4 (789.7/783.2), GSM8K 92.8%, acc 2.339, TTFT p99 1168 ms, TPOT p99 37.9 ms, 128K pass.
  conclusion: Interpretation: +3.8% on T2-002 vs +8.4% on T1-005: part of Triton's gain on T1-005 was in eager mixed steps (FlashInfer planning per step), which T2-001's capture sizes already removed; the decode-step sync removal remains.
validity:     as T1-008 (attention kernel change, exact up to fp reordering, fp8 KV kept) + T2-002's reviews. Capture sizes up
              to 512 with Triton attention: memory checked by the official eval's benchmark + 128K gates.

official #1 (2026-11-14T18:31): 786.4 tok/s at c=16 (mean of 2 trials: 789.7 / 783.2, CV 0.6%) (+23.7% vs BASE), GSM8K 92.8%, MTP acceptance 2.339, TTFT p50 185 ms, TPOT p50 18.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
