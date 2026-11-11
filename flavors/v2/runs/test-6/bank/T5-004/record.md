team:         T5
builds_on:    T5-001 + T5-002 (t5-k3.mb: T1-042 + inert timers + env-gated Marlin 4-m-block tile override AI_M4_CFG)
performance:  unofficial: k=3 + override {64,128,128} + cap 1536 + KV 5.75 GiB = -3.5% vs champion (pooled, same job). Cap 1536 alone: +0.4% (nil).
summary:
  hypothesis: k=3 with the 4-m-block tile override and the highest prefill-graph cap that fits beats T1-042 by ~+1-3% (leader-5 estimate).
  change:     flags B' = num_speculative_tokens 3, prefill grid truncated at 1536 (+ max cap 1536), --kv-cache-memory 6174015488,
              AI_M4_CFG=64,128,128; C' = champion flags with the 1536 grid; A = champion flags (override unset).
  results:    Observed (job 36, 15.3 GPU-min, worker leader-5.w1, agents/leader-5.w1/job5/; built _C verified, gdn_replay_op_available True).
              Cap 1792 at KV 5.75 OOMs in graph capture (8 OOM lines); 1536 boots. 150 prompts c=16, order Bp1 A1 C' Bp2 A2:
              | run | tok/s | TTFT mean/p99 | TPOT mean/p99 | accept |
              | Bp1 k=3 cap1536 ovr | 837.7 | 283/1099 | 17.34/39.3 | 2.701 |
              | A1 champion         | 863.4 | 278/1233 | 16.72/38.2 | 2.345 |
              | C' k=2 cap1536      | 862.3 | 263/1201 | 16.76/37.9 | 2.336 |
              | Bp2 k=3 cap1536 ovr | 818.5 | 309/1318 | 17.63/38.6 | 2.687 |
              | A2 champion         | 853.7 | 267/1223 | 16.72/37.3 | 2.346 |
              B'/A 0.970, 0.959, pooled 0.965. C'/A pooled 1.004. A1 vs A2 differ 1.1%. 0 OOM in served runs; 128K on Bp2 pass (63.5 s), health 200.
              Timers fwd at M=64: 25.92/25.83 ms with override vs 26.32 without (T5-001) => -0.4 ms in situ (microbench -0.76 cold).
              Step 29.55 vs 25.22 (+17%) for E +15% (2.345 -> 2.69).
              Override numerics (random bf16 act + FP8 weights, M=49/56/64, 6 shapes): not bitwise; max-abs 1-4 (0.1-0.5% of max |y|),
              0.1-0.2% of elements differ: K-reduction reorder at bf16-ulp scale. Stock vs stock repeat bitwise equal.
  conclusion: Interpretation: fixed k=3 is a loss on this stack (-1% in job 28 without override, -3.5% here with it; the two differ by
              more than I'd like, so single-pair noise is ~+-1.5%). The timers' model predicts ~-2%; the extra ~1-1.5% is unexplained
              (k=3 runs show higher TTFT: mixed steps also carry the 3rd draft). The cap-1024 "-4%" of T5-001 does not reproduce at
              1536 (C'/A 1.004): the memory price of k=3 via the cap is ~nil, i.e. route B is needed only for K_MAX=4 (16 blocks/req stock).
              The override recovers only half its microbench gain in situ. Plain k=3 / M=64 needs T6's 4-m-block kernel to pay.
validity:     flags + env-gated override (reorder, bounded, measured); k=3 exact spec decode; memory change covered by 0 OOM, 16-req admission,
              128K pass. Not proposed for official eval (slower).
