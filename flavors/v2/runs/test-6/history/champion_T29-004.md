# CHAMPION (run test-6)
record: T29-004 (team T29, leader-29) = T27-002 + checkpoint W8A8 (dynamic per-token-group-128 e4m3 activations vs 128x128 block-scaled FP8 weights; human ruling 2026-11-09) for FP8 block linears at M > 96 only (mixed/prefill steps), via T29's bs_gemm kernel reading the champion's Marlin-repacked weights (per-128-K-block fp32 promotion, fmaf; split-K on K-block boundaries, fixed-order reduction). Decode (M <= 96) bitwise T27-002's Marlin. Default-on (T29_W8A8=2; =0 restores T27-002). +~10 MB/rank; KV 267,721 tokens unchanged.
serve_args.txt: bank/T29-004/serve_args.txt (identical to T27-002's)
official #1: 1045.8 tok/s at c=16 (+8.1% vs T27-002 967.3, +64.5% vs BASE 631.4; trials 1043.9/1047.7, CV 0.26%); TTFT p99 835 ms; TPOT p99 27.4 ms; GSM8K-250 94.4%; MTP acceptance 2.325; 128K pass. bank/T29-004/official/1/results.json
FP8-activation accuracy condition: GSM8K-1319 same job (job 214): W8A8 96.82% vs same code T29_W8A8=0 96.21% (+0.61, within 2.0).
Evidence vs T27-002: both official trials above both of T27-002's (966.3/968.3); same-job ABBA job 207 (T29-003, same code gated) +9.2%, all B > all A.
Promoted 2026-11-09 by the manager.
History: BASE 631.4 -> ... -> T1-042 942.2 (test-5) -> T9-005 948.9 -> T14-004 956.6 -> T27-002 967.3 -> T29-004 1045.8 (test-6, T29).
A performance reference, not a required starting point.
