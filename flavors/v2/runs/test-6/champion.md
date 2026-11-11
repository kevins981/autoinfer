# CHAMPION (run test-6)
record: T50-002 (team T50, leader-50) = T45-005 + T42-003's certified target lm_head (bank/T42-003/delta_vs_T37-002.diff, applied unchanged; VLLM_LMH_CERT=0 disables it) + T50-001's gate_up W8A8 config cfg 103 for M=192..1024 (t29_bs_gemm/configs.json "17408x5120" buckets 192/256/320/768/1024 -> [103,1]). Same W8A8 scheme and M range as before; only the tile shape changes. Details: bank/T50-002/record.md.
serve_args.txt: bank/T50-002/serve_args.txt (= T45-005's: k=3)
official #1: 1181.3 tok/s at c=16 (trials 1179.1/1183.5, CV 0.3%); TTFT p50/p99 142/779 ms; TPOT p50/p99 12.2/27.1 ms; GSM8K 93.6%; MTP acceptance 2.663; 128K pass. bank/T50-002/official/1/results.json
Evidence vs T45-005: both trials above both of T45-005's (1174.7/1172.1), +0.67% cross-job; same-job ABBA job 431 (300 prompts) +1.04%, job 432 (400 prompts, BAAB) all B > all A +1.25%; pooled 8 starts +1.15%, 7 of 8 cross-pairs favour B.
Validity: lm_head part covered by the 2026-11-11 certified-screening ruling (task_definition.md): candidates rescored with the champion's exact chain, startup self-test 0 mismatches on both ranks, fallback to the exact shard. NOTE for the human: T42's proof assumes a tensor-core error bound (2^-11 relative, ~23 fp32 ulps per step; measured 2^-19.4, published behaviour ~2 ulps); the ruling asked for a bound and this is a hardware-model assumption with a 2^8 margin. Manager accepted it; the human may override (then VLLM_LMH_CERT=0 gives the gate_up-only variant).
Promoted 2026-11-11 by the manager.
History: BASE 631.4 -> ... -> T37-002 1142.2 -> T45-005 1173.4 -> T50-002 1181.3 (test-6).
A performance reference, not a required starting point. Bar for a +5% payoff now 1240.4.
