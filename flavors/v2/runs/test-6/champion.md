# CHAMPION (run test-6)
record: T52-005 (team T52, leader-52) = T50-002 + mixed-step all-reduce scheduling, all Python (compiled files byte-identical to T50-002): (1) VLLM_T52_LINKPRI (link priority: chunk 1's DMA ARs wait for chunk 0's on the single D2H copy engine; joined split steps >=256 rows); (2) two-wave layer pipelining for steps >=768 rows (T30's wave code, ported; T52_WAVE=1, T30_WAVE_MIN=768); (3) T32's captured GDN prefill core (T32_GDN_CAP=1) so wave steps are not host-bound. Each flag "0" = off (flags off = T50-002 behaviour). Details: bank/T52-005/record.md.
serve_args.txt: bank/T52-005/serve_args.txt (identical to T50-002's: k=3)
official #1: 1212.4 tok/s at c=16 (trials 1210.1/1214.8, CV 0.3%); TTFT p50/p99 145/667 ms; TPOT p50/p99 11.8/26.6 ms; GSM8K 93.2%; MTP acceptance 2.66; 128K pass. bank/T52-005/official/1/results.json (the official eval runs the defaults, so it covers the flag-flip made after the ABBA)
Evidence vs T50-002: both trials above both of T50-002's (1179.1/1183.5), +2.6% cross-job; same-job ABBA job 477 (400 prompts, A = flags off) all B > all A, +2.82% (B 1213.0/1220.0 vs A 1182.6/1183.8).
Validity: same math up to fp reordering (row-independent GEMMs at a different M; GDN prefill split at 64-aligned positions with the fp32 state handed over; captured core bitwise vs eager; link priority is stream ordering only). FP8 scheme, precision, comm precision, sampling and spec-decode verification untouched; thresholds on step size only. Credit: T30 (waves), T32 (captured core).
Promoted 2026-11-11 by the manager.
History: BASE 631.4 -> ... -> T37-002 1142.2 -> T45-005 1173.4 -> T50-002 1181.3 -> T52-005 1212.4 (test-6).
A performance reference, not a required starting point. Bar for a +5% payoff now 1273.0.
