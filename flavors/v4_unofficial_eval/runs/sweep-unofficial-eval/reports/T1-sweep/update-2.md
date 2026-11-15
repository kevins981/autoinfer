# T1 map update 2 (resync request; HEAD = T2-012; CONFIRMED = T1-005 760.8 tok/s)

## Tested since update-1
| id | record / job | result | verdict |
|---|---|---|---|
| X1 floor | t1-w1 jobs 40-46 | host AR at M=48: write 22 us, read 22 us (each ~90% of PCIe), both at once 30 us, full dependent AR 45-48 us; chunked/warp-specialized/wider-load designs within noise; copy-engine pull not graph-capturable (wait value baked at capture) | AR kernel closed |
| R3 decode GEMM | t1-w1 job 48 | Marlin at M=48 is 1.07-1.14x a cold HBM read floor on every shape; Triton block-FP8 W8A8 slower (2-3.5x on most shapes); only Marlin/Humming/Triton exist on sm_89; L2 capacity does not explain gate_up time | kernel replacement closed; small tile tuning left (T2-012 did it, +0.85%) |
| R4 AR+add+RMSNorm fusion | T1-007 (jobs 52-54) | exact up to 1 ulp; -1.6% (1 start each, separate jobs) | negative as built: fixed 4-block grid (parity safety) gives too little parallelism for the norm. A version with a larger fixed grid (16-32 blocks, microbench showed AR time flat in block count) is untested |
| S1 3 drafts, revisited (sparring T1-1) | T1-008, job 58 | -6.26% (1 start) vs T2-011; acceptance 2.68-2.78 (+14-17%); decode step +13% (Marlin +1.8, host AR+drafter +1.4, GDN +1.1 ms); mixed steps +5% with bigger prefill chunks; median TTFT 270 -> 790 ms, p99 gate fails | negative on current code too; loss concentrated in prefill/TTFT and the tail, not in steady decode per token |

## New findings
- **GPU0 runs at 2235 MHz SM clock under load vs GPU1 2655-2670 MHz** (task/baseline/official/trial1/gpu_clocks.csv; memory clocks equal). Explains the 12% compute-bound skew. Not changeable (root). Software response priced with the sparring partner: overlapping-residency MLP split (GPU1 holds ~4% extra of m, ~0.75 GB; 50/50 in decode, ~46/54 in compute-bound prefill steps; attention/GDN heads can't move because of per-rank state): ~1.5-2%, untried, sizeable.
- Sparring T1-1 beliefs table: sparring/ (T1-1). The claim "within ~1.3x of a decode floor so only <2% wins remain" is unsupported as stated.

## Ranking now (against HEAD T2-012)
1. AR+add+RMSNorm fusion with a larger fixed grid (all calls 16-32 blocks): ~1-1.5%, small effort (kernel exists), needs a same-job A/B.
2. Overlapping-residency MLP split for the slow GPU0: ~1.5-2%, large effort, memory check first.
3. Adaptive draft count by step type (k=3 only on decode-only steps, k=2 when prefill is mixed in): decode per accepted token ~even/slightly better with k=3, so gain is small and uncertain (<~1%); needs engine support (per-step k).

## Addendum (after the resync)
- **The GPU0 clock limit is gone in today's runs.** gpu_clocks.csv of today's unofficial evals (T1-005, T1-008, T2-012) show both GPUs at 2655-2685 MHz under load. Re-running t1-w1's skew analysis on today's profile (job 58, agents/leader-1/skew2_hd.py): prefill-step Marlin GPU0 66.67 ms vs GPU1 66.40 ms per window, i.e. no asymmetry. The 2235 MHz cap was present when task/baseline (nsys 11-06, official 11-07) was recorded.
- Consequence: candidate (2), the overlapping-residency MLP split, is **dropped**: there is no stable asymmetry to balance. Caveat for everyone: the shipped BASE profile/official numbers were taken with GPU0 throttled; comparisons against BASE's 635.9 mix code gains with this environment change (this run's own BASE reference starts, 625.1/639.9, were measured today).
