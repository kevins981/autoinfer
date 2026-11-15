# T2 map, update 1 (resync 1, 2026-11-14)

## Tested since 6-ranking.md
| candidate (my id) | result | evidence |
|---|---|---|
| W4b mixed-step lockstep/CPU (rank 2) | **WON, bigger than napkin**: mixed steps ran fully eager (cudagraph max size 96). Without nsys: mixed = 72 ms + 0.21 ms/row vs decode 38.4 ms. Graph sizes to 512: mixed = 38 + 0.25 ms/row. +6.42% ABBA on T1-004. | T2 jobs 10, 23, 27; T2-001; official T2-002 757.6 (with T1-005) |
| R2 W8A8 Triton block-FP8 GEMMs (rank 1) | **WON once mixed steps were graphed**: no gain on BASE (eager hid it, T2-003), +4.73% ABBA on T2-002, official 788.8 (T2-004), and 819.9 with T1-009's TRITON_ATTN (T2-006). Decode M~45: Triton (tuned BM=16/BN=64-128) ties Marlin; M>=128: 0.6-0.9x Marlin. GSM8K-250 A/B 93.6 bf16 vs 92.4 FP8. | jobs 11, 16, 24, 30, 32; T2-003/004/006 |
| C1 custom host AR (rank 3) | T1 built it (T1-002/004). My PCIe primitives: one direction 22-26 GB/s, both directions at once ~15-17 GB/s each. | jobs 2, 8 |
| W2 L2 prefetch during AR (rank 4) | **LOST**: 0-5.5 us/layer saved vs ~30 napkin; AR +2-4 us. | job 33, T2-005 |
| R2b Marlin tuning (rank 8) | superseded by W8A8 (Marlin no longer used) | |
| K1a GDN DeltaLog (rank 11) | **promoted**: ncu shows 3 full state writes/req/layer (75.8 MB SM->L2 vs 26 MB read), 75 us at 82% DRAM peak; my calibration mismatch (61 us < byte napkin) was L2 holding ~41 MB of dirty writes past the kernel window. Napkin 1.3-3 ms/step. Building. | job 36 |
| chunking prefills to <=512 rows (new) | **not worth doing**: steps >512 rows (14% of time) already GPU-bound (231 ms med at 763 rows vs 233 graphed estimate) | steplog3 analysis, no GPU |
| draft flags, k=3, TRITON_ATTN, AR/GEMM overlap | T1's (T1-001/004/005/007/009) | T1 records |

## Verdicts changed
- Calibration note 3 (GDN under byte napkin) resolved: the kernel window hides drained L2 writes; per-kernel ncu DRAM
  bytes undercount writes. The framework's per-window HBM_bytes needs "bytes written eventually", not in-window.
- The framework's cost model had no CPU resource; the largest T2 win (T2-001) was purely CPU/launch-bound time.
  Ranking it needed a non-nsys measurement (per-step GPU event timestamps), not FLOPs/bytes.
- W8A8's gain was conditional on T2-001: a model that ranks candidates independently misses such interactions.

## Current ranking (on champion T2-006 if promoted, 819.9)
1. GDN DeltaLog: 3.5-8% (napkin, minus replay cost) — building (worker, ws t2-gdn).
2. Fuse per-token-group FP8 activation quant into the preceding RMSNorm / SiLU-mul: ~0.5 ms/step (~1.3%).
3. Decode-M Triton config tuning (M 33-64) / bigger sweep: 1-2%.
4. Reduced-vocab draft lm_head (with FP8): ~2-3% now that FP8 draft head + local argmax landed (T1 owns drafter).
5. Remaining decode CPU idle after TRITON_ATTN: unmeasured (T1 profiling).
Dropped: L2 prefetch (T2-005), PP/DP/DBO (napkin, 4-knobs), chunking >512 rows.
