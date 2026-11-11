# T51 write-up (exploit-long, leader-51): piggyback prefill. Closed on measured kernel limits
Records: bank/T51-001..004. Proposal: proposals/T51-1.md (with updates). Sim: agents/leader-51/sim/{pig.py,pig2.py}.
Worker: leader-51.w1 (agents/leader-51.w1/). Sparring: T51-1. GPU: ~13.6 of 45 granted (jobs 432, 434, 436-439).

## Idea
Prefill is ~28% of the trial: one mixed event per prompt, ~15 ms + 0.166 ms/token (KB111). Decode passes are byte-bound,
so 96-144 prompt rows could ride in every pass and pay mainly the all-reduce. Same as T37's original idea and T45-1's bet.
T45 killed it with a simulator that only tried <=96 rows/pass; with 128-144 rows T45's own model gives +7.4..+9.9%.

## Measured
- T51-001, champion T45-005, budget sweep (job 432):
  - Budget 192: -1.9%. Budget 160: -10.2% vs 2048.
  - Any pass with 1-144 prompt rows costs ~51 ms of wall time, flat in rows; a decode pass costs 24.9 ms.
  - GPU-side excess at ~100 rows (nsys, job 434):
    - GEMM +9.5 ms: all GEMMs switch to bs_gemm, and the AR-window prefetch benefit is lost.
    - AR ~0.07 ms/row.
    - Glue ~5 ms.
  - Wall time is above the GPU span (host enqueue likely, unproven).
  - A calibrated sim reproduces the losses (-3.8 / -11.7% vs measured -1.9 / -10.2%).
- T51-002 (jobs 436/437): in-chain GEMM band with the best existing kernels is 0.07 / 0.09 / 0.07 ms per added row at
  M=128 / 160 / 192. These are steps at m-tile boundaries, not a slope. Config and split-K tuning buy <=3%.
- T51-003 (job 438): 4 CTAs/SM and split-K down do not lower the band, so occupancy is not the limit.
- T51-004 (job 439, ncu): no resource is saturated (L2 55-70%, SM 25-35%). At M>=160 the top stall is math-pipe
  throttle, at ~74% of the e4m3 roof on padded rows. Pad-free one-CTA/SM big tiles are slower.
  - Interpretation: a scheduling-only kernel fix (stream-K) is bounded at a band of ~0.06-0.075 ms/row (estimate, unbuilt).

## Why closed
- Payoff with the achievable band (~0.065 ms/row) requires a fully captured mixed pass with 3 ms of glue: ~+5% in the
  sim. With 5 ms glue it is ~+2.5%. Central +1-2%.
- The captured-pass build is ~40-60 GPU-min and unproven. Below the exploit-long bar.
- Hitting the original +8-11% needs a TP-shape FP8 GEMM above ~85% of the e4m3 roof at M=128-160. No full kernel in this
  run has reached that (KB83/KB98/KB101).
- This is "this attempt failed" for the kernel (no hard limit proven) but a measured ceiling for today's kernel family.

## For future teams
- Rows above 64 in a decode pass cost ~0.07 ms GEMM + ~0.07 ms AR in situ (consistent with KB118/KB120). Prompt rows
  ride at ~0.14 ms vs 0.166 + fixed in an event.
- A captured mixed pass (host floor removed, glue <=3 ms) is the exploit-sized remainder, worth ~+2-5% if piggyback
  rows are added at B~128-144. Today's mixed path has a ~51 ms floor regardless of rows: worth attributing (GPU-idle
  vs host timing) before building.
- Crumbs: qkvz pad-free cfg 134 at M=160 (-5 us/layer); down cfg127+split-K 2 at exactly M=128 (-8 us/layer, chain).
- Tools: T51_STEPLOG GPU-event probe (gpu_worker.py, env-gated), step-log analyzer, nsys category map
  (agents/leader-51.w1/), closed-loop piggyback sim calibrated in situ (agents/leader-51/sim/pig2.py).
- Process: standalone split-K wins did not transfer to the chain; a profiler check (1.1 GPU-min) refuted an L2
  hypothesis before a 14 GPU-min kernel build.
