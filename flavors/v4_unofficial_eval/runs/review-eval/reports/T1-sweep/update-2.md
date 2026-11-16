# T1 map update 2 (closing)

## Tested since update-1
| candidate | result | evidence |
|---|---|---|
| Host-mapped AR (hmar) on T2-004 | **+3.7%, promoted (T1-007, 771.3)**; same kernel measured 3x: 768.6 / 777.2 / 771.3 | T1-005 (job 26), T1-006 (job 30), T1-007 (job 36) |
| hmar robustness fixes (alignment staging, cleanup, UUID P2P agreement, cache key) | 772.4, within noise of T1-007 (start 1 below); no measurable cost | T1-008 (job 41), stress job 40, review T1-4 |
| hmar geometry/unroll sweep (17 variants) | all within ~1 us (M=42: 40.2-41.7 vs NCCL 60.9); no substantial gain found | job 39 |
| hmar write/read overlap (warp-specialized) | both ordered variants slower (51.0 / 53.2 vs 41.4 us at M=42); relaxed publication 31.7 us but wrong results -> publication ordering costs the overlap in this design; deferred | job 42 |
| Pipelined hmar (per-chunk flags) | slower (48.6 vs 41.3 us at M=42) | job 27 |
| k=3 with KV for 16 requests (5.5 GiB) | stopped after 1 start, 764.2 (-0.9% vs 771.3 mean, within T1-007's start range): screen-neutral | T1-009 |
| Async scheduling | already on in the champion (no "disabled" warning; default True for this config) | server.log of T1-007, vllm/config/vllm.py:1815-1849 |

## Verdicts changed
- hmar napkin (-20 us x 135 ARs per step -> +6-8%) delivered about half (+3.7%). The AR count per target graph is 132
  (plus 3 per drafter graph; counters in T1-008/eval/1/hmar_stats), so the shortfall is not the count: part of the AR
  time is overlapped or waiting rather than on the critical path. Framework note: pricing a collective at its kernel
  time over-predicts on both sides (it includes waiting; the saving is not all critical path).
- hmar is bandwidth-limited at ~0.8 us/row, the same per-row cost as NCCL at large M; extending it above 96 rows buys
  nothing (bare NCCL 2.66 MB in ~205 us ≈ 0.79 us/row). The gain is purely NCCL's ~20 us floor at small M.
- k=3 vs k=2 at 16 running requests is roughly neutral once KV is sized for it; not worth the memory.

## Open (untried or unresolved), current ranking
1. Fewer decode all-reduces (merge the attention-output and MLP AR pairs via a rewrite such as sequence-parallel
   RMSNorm, or skip ARs whose consumer only needs a shard): 132 ARs x ~40 us is still ~5 ms of a ~30 ms step.
   Needs graph-level surgery; napkin up to +8% if halved, riskiest assumption = the AR time is on the critical path
   (hmar delivered half its napkin).
2. A correct overlapping hmar (e.g. per-slot flags written by a single thread after a block-wide barrier, not
   per-warp release fences): napkin -10 us/AR -> ~+2-4%; cheapest test = the job-42 microbench harness.
3. Resolve T1-001's unexplained ~3x-napkin gain with matched head-kernel timings (insight for draft-side work).
