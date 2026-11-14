# T56 write-up (sweep, leader-56, framework/). Closed 2026-11-13, no win

GPU: ~11.8 of 100 used (jobs 483, 489, 495, 499), remainder returned. Records bank/T56-001..003. Worker leader-56.w1.
Analysis: reports/T56-sweep.md (inputs, calibration, knob-by-knob sweep, ranking, results in §6). Sparring T56-1, T56-2.

## What the sweep found
- **Calibration** (job 483, fresh nsys of T52-005): the decode step is 24.7 ms.

  | part | ms |
  |---|---|
  | GEMM | 12.3 |
  | 129 ARs | 6.66 |
  | GDN | 2.0 |
  | drafter | 2.06 |
  | small ops + attention | 1.3 |
  | lm_head | 0.75 |

  - The napkin reproduces it within 8%, but only if AR and DRAM time are added: SM-driven PCIe traffic is coupled to DRAM traffic. The framework's max() Window form predicts ~17 ms.
  - Mixed steps are 34% of wall. Their excess over decode is 24% of wall, ~0.167 ms per prompt token at official speed.
- **Ranking** (§4): 1 spatial prefill lane; 2 mid-M FP8 GEMM; 3 drafter fc/embedding table; 4 lower wave threshold; 5 replicated out_proj/o_proj + FP8 all-gather.

## Tests (all negative)

| record | candidate | result |
|---|---|---|
| T56-001 (in-server synthetic lane on T52-005, W8A8 lane) | 1 | decode ITL x1.28/1.39/1.52 at f=32/48/64; 0.137-0.156 ms per lane token GEMM-only vs 0.167 joined; no-go. T57-002 (microbench) found DRAM/L2 arbitration |
| T56-002 (t31b in <=96-row slices vs bs_gemm, M=112-192) | 2 (dispatch only) | +66-84% per layer: no-go. bs_gemm at M=128 is 291 us/GDN layer vs a ~200 us floor |
| T56-003 (S1 of replicated out_proj + FP8 AG; human ruling 2026-11-13 allowed it) | 5 | bitwise premise holds; net -10/+10/+55/+109 us/layer at M=64/128/256/512 (the duplicated GEMM eats the link saving); ~+0.3-0.8% e2e; not built |

Candidate 3 was handed to T57 (with the drafter overhead). Candidate 4 and the mid-M kernel are free; T58 ranks them.

## For later teams
- **Mid-M FP8 GEMM** (warp-specialized weight stream || e4m3 MMA with per-128-K promotion, Marlin layout): ~+1.5-2.2% e2e if it reaches ~1.1x of max(DRAM, MMA) at M=96-512. The weighting is in sparring T56-2. Its attainability is unproven, so it's a kernel project.
- **Replicated out_proj** for M>=512 only: ~+110 us/layer no-overlap. Worth stacking if memory is ever freed for another reason.
- **PP** was measured for decode only (0.91x), not on the full mixed workload.
- **Framework gaps on this machine** (§5):
  - Link and DRAM are coupled for SM-driven traffic.
  - Exact-math ridges are ~76/146 rows, and at M~128 the DRAM and MMA floors coincide.
  - There is no SM-partition resource.
  - Replication's duplicated compute on the critical path has to be priced, not just link bytes.
- **Tools:** agents/leader-56/step_dump.py (one step's kernels from nsys sqlite); vllm/v1/worker/t56_lane.py in workspace t56-lane (phased in-server lane, env VLLM_T56_LANE_CTL).
