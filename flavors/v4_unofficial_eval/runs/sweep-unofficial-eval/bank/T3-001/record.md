team:         T3
builds_on:    T1-016
performance:  "microbench only: no gain (host AR grid 4..32 flat; pipelined variant not better than HEAD's kernel)"
summary:
  hypothesis: HEAD's host-pinned AR (4 blocks) runs 42.5 us at 48 rows and 147 us at ~105 rows in situ (job 153), below
              PCIe speed; more blocks / write||read pipelining, and extending it above 128 rows (NCCL at ~11 GB/s), would help.
  change:     worker t3-w1: VLLM_HOST_AR_GRID (configurable grid), VLLM_HOST_AR_CHUNKS (per-chunk flags, pipelined). Microbench
              agents/t3-w1/bench.py (2 procs, CUDA-graph loop of 64 calls).
  results:    Observed (jobs 159, 161, 164): time ~10 us + 78 us/MB (= write and read serialized at ~25.6 GB/s), flat in grid 4..32
              (64/128 worse). 48 rows 42.7 us, 105 rows 88 us (in situ 147: the rest is rank skew), 256 rows 209 vs NCCL 211,
              350 rows 284 vs NCCL 269. Pipelined kernel: chunks=1 path slower than HEAD's (88 -> 108 us at 105 rows, reproducible);
              best chunks 5-25% better than its own c=1 but not better than HEAD's kernel. All configs bit-exact, ranks equal.
  conclusion: Interpretation: HEAD's kernel sits on the serialized PCIe floor; the 128-row cap is near the NCCL break-even (~256).
              Duplex overlap is the only headroom and the flag/fence cost eats it in this design. Agrees with T1 jobs 40-46.
validity:     no serving change proposed.
