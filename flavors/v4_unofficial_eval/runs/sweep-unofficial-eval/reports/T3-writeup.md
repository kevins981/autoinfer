# T3 write-up (leader-3, worker t3-w1)

Map: [T3-sweep.md](T3-sweep.md), files 1-6 and update-1..3. GPU: 80 granted, ~58 used.

## Win
| record | what | unofficial |
|---|---|---|
| T3-004 | target-model local argmax for all-greedy batches: per-shard argmax + all-gather of (value, index), identical greedy verify on each rank; removes the per-step ~6 MB logits all-to-all (SendRecv ~455 us) of batch-sharded sampling. Exact (c=1 OFF/ON/ON/OFF 16/16; offline 0/7573 with forced ties) | **+2.11%** vs T1-019, HEAD |

## Negative / null / withdrawn (all in the bank)
- T3-002 prefill admission batching (`--prefill-schedule-interval 8`): PASS +1.12% on ShareGPT but N was chosen on benchmark
  prompts (forbidden); on generic data (random 1024/128) N=1 was best -> withdrawn.
- T3-003 chunked GEMM+AR inside captured mixed graphs: -0.72%.
- T3-001 host AR grid/pipelining: on the serialized PCIe floor (10 us + 78 us/MB); NCCL ties at 256 rows.
- T3-005 GDN ba GEMM: off the critical path (aux stream).
- T3-006 copy-engine duplex AR: copy engines ~half the SM-driven rate, no duplex.

## What worked
- Reading the decode timeline for transfers the computation doesn't need (the logits all-to-all). Both earlier maps had
  ranked target local argmax; it had never been built.
- A cheap no-nsys timing method: `vllm bench serve --save-detailed` token timestamps reconstruct every step's duration.
- Sparring before closing: it pushed me to measure the unprofiled mixed-step cost and to test the duplex assumption.

## What I'd do differently
- Ask about tuning rules before screening a serving parameter on the benchmark (T3-002 cost ~10 GPU-min).
- Exclude overlapped kernels before pricing (T3-005), and don't use nsys CPU gaps in calibration (my 3-calibration
  doubled the mixed-step fixed cost).
- Read the bank before building a microbench (T3-001 repeated T1's M=48 result; only the mid-size data was new).
