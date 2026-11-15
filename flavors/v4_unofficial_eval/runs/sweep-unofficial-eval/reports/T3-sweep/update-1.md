# T3 map, update 1 (HEAD = T1-017)

## Tested since 6-ranking
| id | record / job | result | verdict |
|---|---|---|---|
| A1 prefill admission batching (`--prefill-schedule-interval N`) | T3-002; screens 162/165; unofficial #1; generic sweep 174 | ShareGPT: N=8 PASS +1.12% (mean ITL 39.6 -> 37.9 ms). Generic `random` 1024/128 c=16: N=1 341.4 > N=8 334.8 > N=4 330.1 > N=16 327.6 | **withdrawn**: the manager ruled that N may not be picked on benchmark prompts, and on generic data no N>1 wins. The gain is workload-specific (it needs rare admissions relative to steps) |
| B' host AR for mid sizes | T3-001 (t3-w1 jobs 159/161/164) | 10 us + 78 us/MB (serialized at ~25.6 GB/s), flat in grid 4-32; NCCL ties at 256 rows, wins at 350+; pipelined variant no better | **closed** (my napkin assumed a duplex floor the design can't reach; agrees with T1 jobs 40-46) |
| SM3'' chunked GEMM+AR inside captured mixed graphs | T3-003 | unofficial -0.72% (2 starts), 128K pass | **dropped** |
| R7b in_proj/qkv to W8A16/W8A8 | — | T1 (task 32) owns it | handed off |

## Map corrections
- My napkin overpriced A1: the fixed mixed-step cost is lower on T1-017 (W8A8 route), and an empty slot costs about
  1/16 of a step for every held step, more than my N/32 average.
- Window-knob overlaps of the AR (L2 prefetch, chunked AR in graphs) have now failed three ways in this run: the AR's
  PCIe traffic is not an "orthogonal resource" on this box.
- New site found by reading the timeline (G3 -> **RW1**): batch-sharded sampling all-to-alls ~6 MB of target logits per
  decode step (NCCL SendRecv ~455 us on both GPUs, job 153), ~1.6% of the decode step. Greedy-only batches need only
  (max, index) per row. Framework: Rewrite (associativity of argmax over a split vocab index: the max reduction commutes with
  the shard split, and the combine tag is (max, argmin-index)). In t3-w1 task 2.

## Ranking now
1. RW1 target local argmax for all-greedy batches: ~1% of wall (0.45 ms per decode step).
2. SM2 decode column-split GEMM||AR: ~4% napkin, but every AR overlap has failed on this box: low confidence.

## Addendum: RW1 result
T3-004 (target local argmax for all-greedy batches, on T1-019): exact (c=1 OFF/ON/ON/OFF 16/16; offline 0/7573 incl. ties),
SendRecv gone (nsys job 188), **unofficial PASS +2.11%** (863.0/874.5 vs 850.8). Sent to the manager as a champion candidate.
The napkin (0.45 ms of a ~22 ms decode step = ~2%) matched; the worker's 100-prompt A/B (+0.9%) understated it.
