# T1 map update 2 (closing round)
Champion: T1-014, 842.7 tok/s official (+32.5% vs BASE 635.9). T1-015 (T1-014 + row-split for every eager step) did not win: official 837.0.

## Candidates tested since update-1
| candidate (map site) | result | evidence |
|---|---|---|
| k=3 re-test on the new champion (Speculate, token edge) | **screen negative again** -5.8% (1 pair). Extra verify rows cost more than the extra accepted token. k=2 is the optimum for this drafter on this machine | T1-010 |
| fused AR + residual add + RMSNorm (combine-node fixed cost) | +0.26%, mixed, **not significant**; not kept | T1-011, worker job 49 |
| target-side greedy local argmax (Rewrite: (max,idx) distributes over the v split) | **+1.18%** mixed ABBA. Exact for greedy rows; generic fallback whenever a request needs logits/logprobs or non-greedy sampling | T1-012, job 51 |
| row-split microbatching in mixed steps (Window: two row halves, AR of one half under the GEMMs of the other) | **+1.14%**, all B > all A. Works where T1-007 (split d at decode) failed: at >=256 rows the GEMMs are long enough to cover the NCCL AR on a side stream | T1-013, job 55 |
| T1-012 + T1-013 | **official 842.7 (+2.8% vs T2-006)**, champion | T1-014, jobs 57/59 |
| row-split threshold | 160 loses at 129-159 tokens; keep 256 | worker job 58 |
| row-split extended to the eager >=512-token steps (cap 2048), NCCL forced on the side stream | step medians -5.5% (512-799 tok), -17.6% (800-1199), -10% (1600+); <512 unchanged. 1 start/arm +2.6% | worker job 61. End to end: ABBA mixed -0.39% (job 62), **official 837.0 (-0.7% vs T1-014, separate jobs): no evidence of a win** (not an established regression: between-job variation is unmeasured, spar T1-1) | T1-015 |

| draft depth k=1 (spar T1-1: k=2 optimum was assumed) | **screen negative -9.2%** (1 pair): acceptance 1.80 vs 2.36, step only -16% (napkin -17..-20%). k=2 is now measured optimal on both sides | T1-016, job 67 |
| Marlin in-situ gap: is it real? (spar T1-1: ncu vs nsys) | isolated 115 us gate_up reproduced with CUDA events, cold weights, and inside a graph next to 100 MB memcpys (169-173 us per gate_up+down). Not L2, not clocks, not Marlin | T1-017, worker job 68 |

## What changed in the map
- **Window over rows works where Window over d failed.** Same site (AR || GEMM), different split index: splitting rows
  costs nothing extra once rows are past the ridge (compute-bound, Shared-Read Rule cost hidden), and each half's GEMM
  is long enough to cover the other's AR. Splitting d at decode left chunks too short to hide fixed costs. In map terms,
  the Window knob's gain depends on the window length per chunk vs the link's fixed cost, which the framework's
  max(F,BW,Link) does not price.
- **Which AR goes on the side stream matters.** The spin-polling host-memory AR occupies SMs, so it competes with the
  overlapped GEMM. NCCL there is the better choice, even though HOSTMEM is faster serially. This is the same
  "link is not orthogonal" correction as update-1, seen from the other side.
- **Per-step savings didn't carry over end to end** for the >=512 steps (T1-015): measure the full run before trusting step medians.
- **Eager steps >=512 tokens are GPU-bound** (GPU gap > host gap, worker jobs 58/61; T2 found graphing them gains
  nothing). The lever for them is overlap, not launches.

## Remaining in my map (napkin vs T1-014)
- Marlin decode efficiency (0.66 in-situ vs 0.78 isolated, ~3 ms/step, ~+6%): the kernel itself, L2 and clocks are ruled
  out (T1-017). Next suspects: the 2-GPU environment (host-memory AR spinning blocks on the other stream, PCIe traffic,
  power under load) or nsys's timing. Cheapest test: the T1-017 microbench on both GPUs with a concurrent AR loop.
- ReplaySSM (GDN state writes halved): ~+2%, high effort, untried.
- Tree drafts: need a tree-aware GDN verify kernel; untried; worth it only if the step cost per extra row drops.
- Copy-engine all-reduce (the only form where the link is truly a separate resource): untried, ~+1%.
