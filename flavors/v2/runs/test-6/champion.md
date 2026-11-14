# CHAMPION (run test-6)
record: T58-001 (team T58, leader-58) = T57-004 + a one-line default change: VLLM_HOSTAR_L2PF_MSG_MAX 1 MiB -> 2.7 MB, so mid-size host-staged ARs (mixed steps of 97-255 rows) prefetch the next GEMM's weights into L2 like decode ARs do (cache hints only; same mechanism as decode since T1-019; bitwise outputs; no memory change). gate_up 151 -> 117 us/call in <=128-row mixed steps (job 498). Details: bank/T58-001/record.md.
serve_args.txt: bank/T58-001/serve_args.txt (identical to T57-004's: k=3)
official #1: 1229.6 tok/s at c=16 (trials 1231.5/1227.7, CV 0.2%); TTFT p50/p99 136/582 ms; TPOT p50/p99 11.7/25.2 ms; GSM8K 93.6%; MTP acceptance 2.663; 128K pass. bank/T58-001/official/1/results.json
Evidence vs T57-004: both trials above both of T57-004's (1228.3/1222.7), +0.33% cross-job; same-job ABBAAB job 505 (400 prompts, on T57-004) all B > all A, +0.72% (B 1232.1/1236.6/1230.7 vs A 1225.2/1225.8/1221.7). An earlier ABBA on T52-005 (job 503) was +0.26%, mixed. Small but consistent.
Validity: cache prefetch hints only, bitwise outputs, no memory change, serve args unchanged, nothing keyed to the benchmark (a message-size gate).
Promoted 2026-11-13 by the manager.
History: BASE 631.4 -> ... -> T52-005 1212.4 -> T57-004 1225.5 -> T58-001 1229.6 (test-6).
A performance reference, not a required starting point. Bar for a +5% payoff now 1291.1.
