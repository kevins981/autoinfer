# T3 map update 1 (resync 1)

## Tested since 6-ranking (original)
| candidate | result | evidence |
|---|---|---|
| K2-a fused in_proj_ba | dropped: runs on a side stream overlapping qkvz (off the critical path); isolated F.linear 5.8 us | review T3-1, worker jobs 45/46 |
| K4-b column-split GEMM + overlapped AR | dropped: serial 97.3 us vs split 122.6 (latency-bound half AR) / 97.0 (optimistic) | worker job 45/46 (emulated AR) |
| K4-c DBO on mixed steps | dropped: V2 ubatching needs DP>1; hybrid unsupported; HMAR off under ubatching | code reading, review T3-1 |
| **K5-a L2 weight prefetch under each hmar AR** | **win: T3-001 = champion, 816.6 tok/s (+3.7% vs T2-007)**; -10 us per AR+GEMM pair at 16 MiB, >=32 MiB loses | jobs 47, 48, 49; review T3-2 |
| K2-e exact greedy target fast path | screen-positive, unresolved: 823.7 (811.0 / 836.3), +0.9%, start 1 below; NaN/LoRA gaps to fix | T3-002, job 54, review T3-3 |
| K1-a GDN DeltaLog | all 3 slots really written; cold microbench 120/89/60 us for 3/2/1 writes vs 70 us in the server | worker jobs 45/46, code reading |

## Verdicts changed (and why)
- The decode critical path is not the kernel sum: side streams (ba proj) overlap; I now use the union of kernel intervals (idle 2.05 ms).
- Napkin for a prefetch under a link window: the gain is bounded by what the prefetch can finish inside the window (~16 MiB in ~45 us
  with 32 programs) and by L2 capacity (larger prefetch loses). Delivered +3.7% vs ~+3% napkin: in line.
- The logits exchange (ncclSendRecv) is real transport, but removing it buys ~1% (TPOT p50 -0.2..0.3 ms), half my napkin: some of the
  454 us was waiting for the other rank, as T1 warned for collectives.

## New dimensions / framework notes
- L2 is a memory tier the framework's Place knob lacks; a link-bound window (host-staged all-reduce) leaves HBM idle and is a Window
  partner for weight loads of the *next* dependent node (no data dependency on weights). Generalizes to any latency-bound collective.
- Concurrent streams inside one step: the napkin must price the critical path, not the sum.

## Current ranking
1. Per-target prefetch tuning (gate_up vs qkvz/qkv vs MTP; ranges in GEMM tile order) — +0.5-1%, cheap microbench, needs 3 starts.
2. K2-e greedy fast path with the NaN/LoRA fixes — ~+1%, needs 3 starts above the champion.
3. Prefetch under the mixed-step NCCL all-reduces (2.6 MB, ~200+ us windows: room for ~60 MB of prefetch; mixed steps are compute-
   bound so the gain is smaller, but GEMMs at M~260 still stream 12 GB) — unpriced, cheap microbench.
4. GDN DeltaLog — needs a dram-bytes check in the real server.
