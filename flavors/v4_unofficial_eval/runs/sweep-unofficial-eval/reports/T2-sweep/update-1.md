# T2 map, update 1 (after the first wins)

HEAD = T1-005 (confirmed 760.8 tok/s official, +19.6% over BASE); chain BASE -> T1-001 -> T1-004 -> T2-004 -> T1-005.

## Tested since 6-ranking.md
| id | record | result | verdict change |
|---|---|---|---|
| W1 mixed-step CUDA graphs | T2-004 (cap 512) | unofficial PASS +4.49% vs T1-004; **confirmed** 728.8 tok/s official | won; half the 8-10% napkin (steps >512 tokens stay eager; part of the measured idle was likely nsys). Cap 2048 OOMs at capture next to the FP8 draft head copy (T2-002). |
| PL1 L2 prefetch during AR | T2-005 (on T2-004, NCCL AR) | PASS +2.30% | — |
| | T2-006 (on T1-005, host AR, 32 MB/16 CTAs) | negative (1 start) -1.82% | **parked**: the host one-shot AR is more sensitive to concurrent HBM/SM traffic; |
| | T2-007 (24 MB/12 CTAs) | negative (1 start) -1.54% | the gain was an AR-window effect specific to NCCL LL |
| KR1 GDN 1-write state | worker t2-w1 | upper bound measured: -51 us/layer chain (job 16); kernel bitwise-exact in unit test; server corrupts state for some requests (layers 17/40, not 0); debugging (bounded: +4 GPU-min) | prefix caching must stay ON (human ruling) -> needs align-mode support |
| SP1 3 drafts | T2-008 | running | |

## Microbenchmark lessons (framework)
- A "Window: HBM ‖ Link" overlap must be priced with the link's own HBM/SM footprint: the sleep-kernel stand-in (job 6) predicted -30 us per GEMM, real NCCL (job 11) showed the AR slowing by as much at 62 CTAs; at 16 CTAs the net was positive with NCCL (job 19, T2-005) but negative with a host-memory AR (T2-006/007). The framework's concurrent-window max(F, BW, Link) assumes independent resources; on this box the link IS SM + HBM traffic (GPU stores/loads to mapped host memory), so Link and HBM are coupled.
- Deferred writes (GDN state) cost the following kernels ~11 us/layer (job 16): confirms 3-calibration's "deferred write drain" note.

## Current ranking (napkin vs HEAD T1-005)
1. SP1 3 drafts: 2-5% (running).
2. KR1: up to ~5% if made correct with align mode; high integration risk.
3. W4 drafter attention backend with in-place metadata update (TRITON_ATTN for the drafter): ~1-1.5%.
4. W1b: cap max_num_batched_tokens at 512 so every mixed step is graphed: ~1%.
5. RW1 target local argmax: largely covered by T1-004 (batch-sharded sampling).
