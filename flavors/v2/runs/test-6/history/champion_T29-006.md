# CHAMPION (run test-6)
record: T29-006 (team T29, leader-29) = T29-004 + three exact items, bitwise equal to T29-004's numerics: (1) v2 bs_gemm tiling table (split-free; 0 diffs vs T29-004's table on live activations, 78 M values); (2) W8A8 activation quant fused into its producers at M>96 (Gemma add+RMSNorm -> quant at M>=256; silu_mul -> quant at M>96), bitwise on 147 served-activation dumps (y, residual, fp8 bytes, scales; all layer types, M 3..2048); (3) GDN glue trim in mixed steps (in-server self-check 192/192 bitwise). Decode (M<=96) bitwise T27-002 Marlin. Switches: T29_FUSED=0, T29_GLUE=0, T29_W8A8=0.
serve_args.txt: bank/T29-006/serve_args.txt (identical to T29-004's / T27-002's)
official #1: 1062.1 tok/s at c=16 (+1.6% vs T29-004 1045.8, +67.0% vs BASE 631.4; trials 1062.3/1061.8); TTFT p99 851 ms; TPOT p99 30.3 ms; GSM8K-250 93.6%; MTP acceptance 2.33; 128K pass. bank/T29-006/official/1/results.json
FP8-activation accuracy condition: GSM8K-1319 reused from job 214 (T29-004: 96.82 vs 96.21 bf16 ref) under the bitwise-reuse addendum (task_definition.md).
Evidence vs T29-004: both official trials above both of T29-004's (1043.9/1047.7); same-job ABBA job 231 (A B B A) +1.09%, all B > all A.
Promoted 2026-11-10 by the manager.
History: BASE 631.4 -> ... -> T1-042 942.2 (test-5) -> T9-005 948.9 -> T14-004 956.6 -> T27-002 967.3 -> T29-004 1045.8 -> T29-006 1062.1 (test-6).
A performance reference, not a required starting point.
