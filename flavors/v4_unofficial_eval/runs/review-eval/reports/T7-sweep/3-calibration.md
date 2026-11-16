# T7 sweep, step 3: napkin cost model, calibrated (per GPU, one full-batch decode step, M = 16 x 3 = 48 rows)
Measured side: the champion's own nsys node profile (bank/T5-001/record.md notes, job 118, 64 prompts, full-batch decode steps D>=15;
nsys adds some overhead) and the harness run (bank/T5-001/eval/1). T7 had no GPU time to re-profile before choosing (run ends 08:15).

| Part | Napkin (bytes / BW or latency) | Measured (job 118) | Ratio |
|---|---|---|---|
| gate_up GEMM (64 x 5120 x 8704 x 2 B fp8 = 5.7 GB) | 5.7 ms | 7.08 ms | 0.80 of BW |
| down GEMM (2.85 GB) | 2.85 ms | 3.32 ms | 0.86 |
| GDN in-proj qkvz (48 layers) | ~1.6 ms | 1.94 ms | 0.82 |
| out/o_proj (64 layers) | ~1.1 ms | 1.32 ms | 0.83 |
| All-reduce (HMAR), 128 x 48 rows x 10 KB = 0.5 MB/dir each | 20 us x 128 = 2.6 ms at 25 GB/s | 5.96 ms (~46 us each) | 0.43: latency-dominated (host round trip), not bandwidth |
| GDN recurrence (DeltaLog): read 1.2 GB fp32 state | 1.2 ms | 1.72 ms | 0.70 |
| target lm_head (1.27 GB bf16) | 1.3 ms | (in "rest") | |
| draft: 2 MTP steps (layer 0.2 GB + fp8 head 0.64 GB) | ~1.7 ms | head 0.78 ms/step at M=16 (T3 mb4a, in champion history) | |
| attention (16 layers, short contexts), norms, conv, quant, sampling, launches | ~1.5-2 ms | remainder | |
| **Sum** | ~19 ms at 100% BW, ~23 ms at the measured efficiencies | **29.0 ms wall** (node trace) | the ~4-6 ms gap = launch/sync gaps + nsys overhead + small ops |

Whole benchmark: 400 x 228 / 895 = ~102 s. Decode steps ~ 91.2K / (16 x 2.343) ~ 2,430 x ~27 ms (TPOT p50 16.2 ms x 2.343 = 38 ms per
request-step incl. mixed steps) -> decode ~65-70 s, mixed/prefill steps the rest (~30-35 s; materialize ~1.5 ms per mixed step).
Interpretation: the napkin reproduces the GEMM parts within 15-20% (Triton W8A8 at ~80-86% of DRAM BW) and the GDN part within 30%;
the AR is the one part the bandwidth napkin misses by 2x: it is a latency cost (~46 us x 128), so it must be priced per call, not per byte.
Per-step cost is ~75% fixed (weights) w.r.t. rows, which is what makes "more rows per step" (Speculate) attractive.
