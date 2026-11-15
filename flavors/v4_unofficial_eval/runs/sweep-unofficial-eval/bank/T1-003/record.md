team:         T1
builds_on:    T1-001
performance:  unofficial negative (1 start, -4.89%); TTFT p99 gate failed
summary:
  hypothesis: with T1-001 the draft step is cheaper (fp8 head ~0.7 ms, no logits all-gather), so a 3rd MTP draft may pay: tokens/step 2.35 -> ~2.7 if 3rd-position acceptance ~0.35-0.40, vs step cost +~4-5 ms (M 48->64: Marlin +~1, AR +~2.7, GDN +0.6, 3rd draft step ~1.2). Napkin: roughly +0-6% (T1 sweep S1).
  change:     serve args only: T1-001's speculative config with num_speculative_tokens 3.
  results:    Observed: unofficial #1 stopped after 1 start: 621.2 vs HEAD ref mean 653.1 (-4.89%). Acceptance length 2.708 (HEAD 2.353); per-position 0.77-0.78 / 0.54-0.55 / 0.375-0.39 (napkin assumed p3 ~0.40: right). p99 TTFT 2060 ms (gate 1609: FAIL), TPOT p99 43.2. The '128K FAIL' line reflects the early stop (no long_request.json was produced), not a measured failure.
  conclusion: Interpretation: +15% tokens/step did not cover the step cost: implied step time +~21% (M 48->64 raises AR bytes 33% on the slow link, plus prefill-containing steps get heavier -> TTFT p99). k=3 needs much cheaper ARs/steps first; revisit after a faster all-reduce. A 1-start negative, not a measured fact.
validity:     Exact speculation unchanged (greedy verification against the target); only the draft count changes, which TASK.md lists as open. Memory: more draft slots/graph sizes -> run with --long. Latency gate checked from the eval's p99s.

unofficial #1 (2026-11-15T05:23): negative (1 start): -4.89% vs HEAD T1-001 (B 621.2 vs reference mean 653.1 over 2 starts, 0 run in this job; 200 prompts); 128K FAIL, p99 TTFT 2060 ms / TPOT 43.2 ms, MTP acceptance 2.708; gates failed: latency p99_ttft_ms 2060 > 120% of BASE 1341, 128K request (unofficial/1/unofficial.json)
