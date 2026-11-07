# CHAMPION (run test-3)
record: T1-005 (round 1): host-staged 2-rank all-reduce + all-gather (code default on 2-rank TP
without P2P) + serve flag --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'.
official #1: 699.2 tok/s at c=16 (+10.7% vs BASE; 3 trials 700.7/700.8/696.2); TTFT p50 210 / p99 1309 ms;
TPOT p50 20.9 / p99 50.6 ms; GSM8K 93.2%; MTP acceptance 2.339. bank/T1-005/official/1/results.json
Gates (vs BASE): TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%. All pass.
Previous: BASE 631.4 tok/s (bank/BASE). Profiles of T1-005: in-server nsys under leader-1's T1-003/T1-005 records (see bank).
A performance reference, not a required starting point.
