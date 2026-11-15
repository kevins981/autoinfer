team:         T1
builds_on:    T1-001
performance:  "screen negative (1 pair, -3.64%)", T1 GPU job (t1-k3), agents/leader-1/jobs/k3/
summary:
  hypothesis: With the cheaper drafter of T1-001, a 3rd draft token adds acceptance (~+0.4) for ~+5 ms/step (napkin:
              +16 verify rows ~+2.7 ms incl. AR growth, +1 draft step ~1.9 ms) -> ~+4%.
  change:     num_speculative_tokens 2 -> 3 (flags of T1-001 otherwise).
  results:    Observed: A (T1-001) 651.6 tok/s, acc 2.362; B 627.8 tok/s, acc 2.735 (per position 0.774/0.558/0.403);
              B TTFT p99 2079 ms (gate: 1.2 x 1341 = 1609 ms) / TPOT p99 52.0 ms. Stopped after 1 pair (stop-loss).
              128K request on the k=3 config: pass (53.1 s, server answers afterwards).
  conclusion: Interpretation: acceptance matched the napkin (2.74 predicted), but the step cost grew ~20% (2.735/2.362
              x 651.6/627.8), ~1.5x the napkin's +13%: the extra verify rows cost more than the Marlin/AR model says
              (Marlin at M=64 vs 48, GDN with 4 rows, mixed steps), and TTFT p99 breaks the latency gate. Revisit only
              after the per-step fixed costs drop (faster AR, reduced draft vocab), and check TTFT p99 first.
validity:     exact (draft count only); not a candidate (latency gate).
