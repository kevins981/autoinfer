# T1 map update 1 (first resync)

## Tested since 6-ranking.md
| candidate | record / job | result | verdict change |
|---|---|---|---|
| S-K3 MTP 3 drafts | T1-001, job 6 | 602.5 tok/s (1 start, stopped), acceptance 2.701 (as napkinned), p99 TTFT 2134 / TPOT 62.7 ms (gates fail) | step cost +22% vs napkin +15%: the extra draft step and the 64-row verify are dearer than priced. Now conditional on cheaper draft steps (T2's FP8 / reduced-vocab draft head, local argmax) and cheaper ARs; dropped for now |
| I-AR (a) NCCL protocol | T1-002 (promoted), env-variant + record evals | 663.2 / 660.0 and 662.3 / 655.4 vs T2-001 648.6 / 658.6: +1.0% | win, but ~1/3 of napkin. NCCL picks LL for every size inside the engine; Simple is 1.3-2x faster standalone (leader-2.w1 table) |
| I-AR (b) pinned-host custom AR | T1-003, jobs 13, 17 | exact; standalone graph-replay 52-92 us @48 rows vs NCCL Simple 57; loses at >=96 rows | demoted: the latency floor of host-flag spinning (~20-40 us) is close to NCCL Simple at these sizes. Parked |
| I-AR (c) flashinfer PCIe IPC AR | jobs 3, 10 | fails to initialize on this machine (no P2P) | dropped |
| W1 graphs for mixed steps (T2) | T2-001 | +2.3% (napkins: T2 +12%, mine +6%) | the skew/launch part was smaller than both estimates |

## Corrections to my calibration
- Standalone NCCL Simple at 260 rows: 203 us (2.66 MB, 13 GB/s tensor throughput, ~PCIe both ways via host); the host path's asymptote is ~13 GB/s of tensor bytes, not 25: each byte crosses PCIe twice per rank (up to host, down to the peer). My napkin Link_BW for an all-reduce through host memory should be ~13 GB/s of payload, which halves my I-AR ceiling: prefill AR napkin 135 x 205 us = 28 ms (not 15), decode 135 x ~40-57 us.
- Framework note: "Link_BW" for a no-P2P pair is a two-hop path (GPU->host->GPU); the window formula should count link bytes on both hops.

## Current ranking (T1 lane)
1. R-ARGMAX target side (~0.95 ms/decode step, ~+2%), in progress (worker, CPU-side).
2. W-DBO in mixed steps (hide ~28 ms of AR transfer per prompt step under the other half-batch's compute-bound GEMMs): ~+3-6%. Needs a 2-stream microbenchmark first.
3. W3 AR || producing GEMM by output-column chunks (decode): ~+3-5%, high effort.
4. S-K3 again once draft steps are cheaper.
