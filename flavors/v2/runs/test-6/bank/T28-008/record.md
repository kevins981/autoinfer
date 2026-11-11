team:         T28
builds_on:    T29-004 + T28-004 (Marlin-e4m3 at M<=96) + T29's fused π-quant producers (t29-fu, silu-only, ALL_M) ; k=2, two-slot off
performance:  "unofficial same-job ABBA vs T29-004 (job 234, 300 prompts): LOSS, all B < all A, -1.84% (B 1015.7/1012.5 vs A 1031.2/1035.2). Decode step +0.65 ms (ITL med 24.94 -> 25.59), E unchanged 2.33."
summary:
  hypothesis: at k=2 (M=48) the e4m3 kernel (standalone -0.33 cold / -1.0 warm ms/forward vs Marlin) with the fused silu quant (one launch fewer per down GEMM) beats the champion's bitwise Marlin decode by ~+1-2%.
  change:     workspace t28-comb.b = T29-004 + Marlin-e4m3 (T28-004/007 table: M=48 rows single-slab tmb=3 per shape) + two-slot GDN (off here) + T29's fused producers port (w2; QuantizedActivation via _input_quant_key; T28_E4M3_CHECK_QUANT one-time diff log). B env: T28_E4M3=1 T29_FUSED=1 T29_FUSED_ALL_M=1 T29_FUSED_NORM=0; A env: T28_E4M3=0 T29_FUSED=0. Smoke (32 prompts): CHECK_QUANT max|dq bytes|=0, max|d scale|=0 for M=1..1879 on both ranks.
  results:    Observed (agents/leader-28.w2/job1/out8.0026):
    | start | tok/s | E | TPOT p50/p99 | ITL med | TTFT p50/p99 | E/ITL |
    | 1 A | 1031.2 | 2.331 | 14.04/29.1 | 24.96 | 148/855 | 0.0934 |
    | 2 B | 1015.7 | 2.327 | 14.24/25.0 | 25.60 | 146/764 | 0.0909 |
    | 3 B | 1012.5 | 2.331 | 14.30/27.7 | 25.57 | 159/843 | 0.0912 |
    | 4 A | 1035.2 | 2.331 | 13.98/35.3 | 24.92 | 161/817 | 0.0935 |
    Means A 1033.2 / B 1014.1; decode tokens/ms -2.7%. e4m3 line in both B starts, in neither A; KV pool 267,721 in all (two-slot off); 0 OOM/preempt. 9.7 GPU-min.
  conclusion: Interpretation: in situ at M=48 the e4m3 path costs +0.65 ms/step over Marlin bf16 even with the fused silu quant; the standalone cold/warm advantage does not transfer (consistent with T28-006/007: the e4m3 table's configs re-stream B through L2 where Marlin's single tmb=3 slab does not, and the ~0.35 ms of remaining quant launches). Not separated: fused silu quant vs kernel. Together with T28-005 (k=3 null) the Marlin-e4m3 decode kernel is a loss at both k on this stack. Direction closed.
validity:     not a candidate. (Math: FP8 activations per the ruling; fused silu quant bitwise vs the standalone quant, logged.)
