team:         T2
performance:  "OFFICIAL #1: 699.5 tok/s (697.7/701.3), +1.2% vs champion T1-003 (691.3), +10.0% vs BASE; all gates pass. Controlled ABAB same-job A/B: +0.2% vs T1-003 = within noise"
summary:
  hypothesis: on the host-AR champion, the exact greedy target fast path (T2-004) beats batch-sharded sampling: it removes
              the sharded path's SendRecv (0.41 ms/step) and sampling glue; estimate -0.5 ms of a ~34 ms step = +1.5%.
  change:     T1-003 diff + T2-004 diff (no overlap; content identical to the concatenation). serve_args = T1-003's
              speculative-config WITHOUT --enable-batch-sharded-sampling (the fast path runs only in the non-sharded path).
              Builds on T1-003 and T2-004.
  results:    Observed (job 21, leader-2.w2): CHAMP 661.8/668.2, G 675.9/681.0 tok/s; TTFT p99 CHAMP 1378/1344, G 1473/1370;
              TPOT med ~21.0 both; acceptance 2.34-2.35 both; gsm8k-50 88.0% (=BASE). Fast-path log line on both ranks.
              nsys graph-level: target-end -> next-target window 5.90 vs 6.40 ms (-0.50 ms, as estimated); inter-graph GPU idle
              unchanged 1.60 ms. Caveat: no second CHAMP server, order drift not controlled.
  conclusion: Interpretation: real ~+1.5-2% from removing the sharded-sampling exchange; with the host AR the balance
              tipped vs round 1 (where the two were equal).
  ABAB (job 26, leader-2, order-controlled, fresh server per leg, bench 200 c=16; agents/leader-2/abab/):
              tok/s A1 CHAMP 657.8 | B1 T2-006 661.2 | A2 CHAMP 661.4 | B2 T2-006 661.1. Means 659.6 vs 661.2 (+0.2%).
              p99 TTFT 1209/1226/1505/1226, p99 TPOT 51.1/47.3/61.6/52.3, acceptance 2.345/2.344/2.355/2.353.
              Logs confirm arms: CHAMP sharded=on fast-path=off; T2-006 sharded=off fast-path=on; host AR on in all.
              Interpretation: the earlier +2% (job 21, one server per arm, order CHAMP then G) was server-to-server drift;
              the fast path and batch-sharded sampling perform the same with host AR too (as in round 1). The official
              +1.2% is within cross-job noise (T1-003's own trials differ by 1.4%).
validity:     leader-2: T1-003 part reviewed/officially passed; T2-004 part reviewed in round 1 (exact argmax, lowest index on
              ties = baseline tl.argmax; greedy verify == rejection_sample at T=0; generic condition on request params;
              falls back to the full path otherwise). Flags: drafter-only options ruled allowed. OK.

official #1 (2026-11-07T06:34): 699.5 tok/s at c=16 (mean of 2 trials: 697.7 / 701.3, CV 0.4%) (+10.0% vs BASE), GSM8K 92.8%, MTP acceptance 2.335, TTFT p50 225 ms, TPOT p50 21.0 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
