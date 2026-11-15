# T3 sweep, step 6: promising candidates, ranked

Reference: champion T2-008, official 880.1 tok/s (103.3 s). Re-profiled (job 80, nsys): decode step span 31.1 ms
(body GEMMs 16.7, AR 6.4, lm_heads 3.1, GDN 1.8 + 0.2 conv, small kernels ~1.8); ~2490 decode steps ≈ 75 s, mixed
steps ≈ 28 s. Mixed steps: AR exposed 18-30 ms in 100-250-row steps (no row-split below 256), 11-15 ms in 256-512,
32-77 ms in eager >512 (job 80, `exposed.py`).

Tested since 4-knobs (all mine, T3): T3-001 hostmem AR for all sizes: **screen negative -6.6%** (job 79).
fp16 accumulation within a K-block (allowed by the contract): Triton emits `mma...f16.e4m3.e4m3.f16` on sm_89, but
32-128-term sums of max-scaled e4m3 products overflow fp16 (sigma of a 32-term partial ~24-85K vs 65504): **dead**
(no GPU used).

| # | candidate | napkin gain (total) | wins when | riskiest assumption | cheapest test |
|---|---|---|---|---|---|
| 1 | **AR-window L2 work**: extra blocks in the hostmem AR kernel prefetch the next GEMM's weights (evict_last) into L2 during the ~48 us PCIe-bound AR; as a side effect the GDN state's dirty lines are written back in the HBM-idle window instead of during gate_up | decode: gate_up after GDN 132 -> ~100 us, after attention 113 -> ~100, qkvz/qkv partly from L2: **~1.5-3 ms/step = 4-7%** | the prefetch fits inside the AR's time without slowing the AR (AR uses 32 CTAs) and the lines survive the norm kernel | in situ, L2 replacement keeps the prefetched lines (T2-005's microbench saved only 0-5 us; mine (job 82) saved 38-68 us when the prefetch had time) | 2-rank microbench AR+prefetch, then ABBA with an env switch (**worker leader-3.w1, running**) |
| 2 | Copy-engine all-reduce for the row-split side stream in mixed steps (memcpy D2H/H2D + tiny flag kernels; no SMs spinning), then row-split down to ~96 rows | mixed-step exposed AR ~9 s total -> maybe half: **~3-4%** | CE AR at ~22 GB/s doesn't slow the overlapped GEMMs, and its fixed cost (~15-20 us) stays below the GEMM per half | that SM contention, not rank skew, is what makes the side-stream NCCL slow (job 80: 230 us mean skew) | 2-rank microbench: Triton GEMM at M=256 alone vs concurrent with a CE AR vs with NCCL |
| 3 | DeltaLog with a 2-step checkpoint (write the state every other step) | ~0.6 ms/decode step + less drain: ~1.5% | the replay of up to 6 logged rows stays cheap | interaction with T2's materialize hooks (prefix-cache boundaries, mixed steps) | build in T2's kernel; bitwise test; ABBA |
| 4 | Reduced-vocab draft head with a better generic frequency source | draft head 1.56 ms/step -> ~0.45: ~2.5% minus acceptance loss | target tokens outside the subset < ~1% in chat text | a non-benchmark frequency table that matches chat text (tokenizer id order gave 3.2% OOV at 64K, T1-006) | offline coverage check on generic text (no GPU) |
| - | fused norm+quant; decode-M Triton configs (gate_up wave quantization: 408 CTAs = 3.2 waves at 1 CTA/SM) | ~1.5% / ~2-4% | | | **T2 owns these** (agreed with leader-2) |

Dead or done: hostmem AR for large messages (T3-001), fp16 accumulate, k=3/k=1/trees (T1-003/010/016), split-d
chunked overlap (T1-007), PP and placement variants (all maps), chunk size <=512 (T2 analysis, T1-015).
