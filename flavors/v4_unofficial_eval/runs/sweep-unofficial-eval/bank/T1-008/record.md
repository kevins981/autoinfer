team:         T1
builds_on:    T2-011
performance:  unofficial negative (1 start): -6.26% vs T2-011; TTFT p99 gate failed
summary:
  hypothesis: revisit 3 MTP drafts (T1-003: -4.89% on T1-001, 1 start) on the current champion: drafts are now cheap (fp8 + reduced-vocab head, local argmax), mixed steps are graphed (T2-004), small ARs are faster (host AR). Acceptance 2.35 -> ~2.71 (+15%) measured in T1-003; break-even needs the step cost increase below ~15% (was ~21%). Prompted by sparring T1-1 (the old negative was 1 unprofiled start on stale code).
  change:     HEAD (T2-011) code and serve args with num_speculative_tokens 3.
  results:    Observed: 686.6 vs HEAD ref 732.4 (-6.26%), 1 start (early stop). Acceptance 2.680 (HEAD ~2.35; +14%), per-position 0.762/0.528/0.380. p99 TTFT 1910 ms (gate 1609: fail), TPOT p99 42.0.
  conclusion: Interpretation: tokens/step +14% but the step got ~+22% slower even on the current champion, so the sparring question 'stale code?' is answered: still negative, worse than on T1-001. The large TTFT p99 rise points at mixed (prefill-containing) steps, not only decode. Decomposition (t1-w1 job 58, nsys 64 prompts, k=2 vs k=3 on T2-011 code, agents/t1-w1/hd/): decode step 34.98 -> 39.57 ms (+13.1%: Marlin +1.8, host AR+drafter +1.4, GDN +1.1 ms) for acceptance 2.38 -> 2.78 (+17%); mixed steps ~+5% (bigger prefill chunks 231 -> 314 median tokens); median TTFT 270 -> 790 ms. So per accepted token decode is roughly even to slightly better with k=3; the loss concentrates in the prefill/TTFT phase and the run's tail. Not pursued further (budget).
validity:     exact speculation unchanged (greedy verification against the target); draft count is open in TASK.md. Memory (graph sizes, draft slots) -> --long.

unofficial #1 (2026-11-15T07:23): negative (1 start): -6.26% vs HEAD T2-011 (B 686.6 vs reference mean 732.4 over 2 starts, 0 run in this job; 200 prompts); GSM8K/128K not run (stopped after 1 start), p99 TTFT 1910 ms / TPOT 42.0 ms, MTP acceptance 2.680; gates failed: latency p99_ttft_ms 1910 > 120% of BASE 1341 (unofficial/1/unofficial.json)
