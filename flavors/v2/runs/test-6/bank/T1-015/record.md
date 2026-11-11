team:         T1
builds_on:    T1-013
performance:  "unofficial: +0.1% at c=14 (k=3 vs k=2), null result; gate for R3-B v2 -> NO-GO"
summary:
  hypothesis: k=3 lost in T1-008 only because 14/16 requests fit (KB13); at c=14 (both fit) k=3 with replay should show its
              intrinsic gain (est +3..8%), which would justify R3-B v2 (shrink GDN spec blocks so 16 fit).
  change:     none in code (T1-013); serve num_speculative_tokens 3; bench at --max-concurrency 14. Worker leader-2.w8, job 53.
  results:    Observed (agents/leader-2.w8/out/gate14, A B B2 A2, 200 prompts c=14): k=2 678.7/680.5, k=3 680.3/680.1 tok/s;
              acceptance 2.33 -> 2.65/2.69; TPOT p50 flat ~17.8-18.1 ms; k=3 Running 14 / Waiting 0 throughout; replay on
              at k=3; KV 259,441 vs 267,721 tokens. Estimate +3..8%, measured +0.1%.
  conclusion: Interpretation: +14.5% acceptance is cancelled by ~+14% step time (Marlin M 42->56, extra draft pass, GDN 4 tok,
              one more verified position). Fitting 16 at k=3 (v2) would only remove the stall, not add gain. Don't build v2.
              Caveat: c=14 not 16, FlashInfer backend (pre-TRITON), 2 trials per arm, step time not profiled.
validity:     no code change; spec decode stays exact. Not for official eval.
