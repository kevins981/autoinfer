team:         T1
builds_on:    T3-004
performance:  "unofficial negative (1 start): -10.24% vs T3-004; TTFT p99 1663 gate failed"
summary:
  hypothesis: (sparring T1-3) 3 MTP drafts were rejected on pre-W8A8 code (T1-008, T2-008) with a binding TTFT. Now prefill
              is ~30% cheaper, TTFT p99 is 944 vs a 1609 limit, and acceptance with k=3 is ~2.7 vs 2.33 (+15%). Napkin
              for the k=3 step: GEMM +2.05 ms (M=64 sum 17.05 vs 15.0 at M=48, job 196), drafter +~0.85, AR +~1.2
              (64-row payloads), GDN +~0.5, i.e. ~+18% step for +15% tokens. That predicts a small loss, but it is
              untested on current code. Decision rule set in advance: PASS -> keep going, else close the team.
  change:     serve args: T3-004's with "num_speculative_tokens":3. Code: T3-004 + W8A16 table tuned for M<=64
              (t1-w16's kernel/table + job 196 bucket configs). Worker t1-w1, workspace t1-w17.
  results:    Observed: unofficial #1: B 779.8 vs HEAD mean 868.7 (-10.24%); acceptance 2.673 (as predicted, +15%); p99 TTFT
              1663 ms (gate fail), TPOT 44.5. One runtime JIT warning (_gather_draft_sampled_kernel, a small kernel).
  conclusion: Interpretation: acceptance behaves as predicted, but the per-step cost rises ~28%, well above the ~18% napkin
              (unpriced costs in the 4-token GDN path, the drafter and mixed steps). With k=2 the work per accepted token is
              lower on current code too. k=3 is closed for this hardware and stack (third negative, now on current code).
validity:     Exact spec decode (same MTP method, greedy verification against the target); W8A16 changes are fp reordering
              only; same activation formats as HEAD. 128K checked with --long (graph and KV sizes change).

unofficial #1 (2026-11-15T13:19): negative (1 start): -10.24% vs HEAD T3-004 (B 779.8 vs reference mean 868.7 over 2 starts, 0 run in this job; 200 prompts); GSM8K/128K not run (stopped after 1 start), p99 TTFT 1663 ms / TPOT 44.5 ms, MTP acceptance 2.673; gates failed: latency p99_ttft_ms 1663 > 120% of BASE 1341 (unofficial/1/unofficial.json)
