# T4 sweep, step 6: promising candidates, ranked

Reference: champion T3-009, official 951.5 tok/s = 95.9 s for the 400 prompts. The calibrated decomposition is in 3-calibration.md.
After the comparison (5-compare.md), every decode-step candidate on my map is either built or measured dead by T1-T3.
What's left is in the **mixed steps** (42 s of 96, per my nsys-based calibration).

| # | Candidate | Napkin gain (wall) | Wins when | Riskiest assumption | Cheapest test |
|---|---|---|---|---|---|
| 1 | **CUDA-graph capture sizes past 512** (add 576..1024, 8 sizes) so the 76 prompts with P+45 > 512 run graphed instead of fully eager | ~60-75 ms x 76 = **4.5-5.7 s, +5-6%** | eager >512 steps carry CPU idle that a graph removes (graph trace: 30-60 ms idle per ~210 ms step) and graph memory fits | that >512 steps are not already GPU-bound (T2's verdict, Marlin-era slope) and that 8 more sizes don't OOM (T2's 83-size attempt did) | serve-flag ABBA at 400 prompts + 128K request (**E1, job 145, running**) |
| 2 | **Small mixed steps (P+45 < 256, 232 prompts): take the eager GDN core off the CPU's critical path**, e.g. a per-layer CUDA graph of the mixed GDN core keyed on (padded prompt-token bucket, number of spec decodes), with persistent metadata buffers; padding rows with g=0, beta=0, k=v=0 leave S exactly unchanged | if the step is ~75 ms vs a ~35-40 ms GPU floor: **~8-10 s, +9-11%**; if T2's "2.5 ms real gaps" holds: <1% | the CPU time of ~30 eager launches per GDN layer (~1.4 ms) is really exposed without nsys | the nsys reading (PROFILING.md: nsys inflates eager launches) | non-nsys per-step GPU-event log (T2's steplog, ws t4-steplog) in the same E1 job: median step time of <256-token mixed steps vs decode steps |
| 2b | fallback for 2: merge small prefills (admit pending prompts together) | up to ~half of 2's gain; TTFT p99 up | 2 is real but too costly to build | TTFT p99 gate (1609 ms) | scheduler knob after 2's measurement |
| 3 | Mixed-step spec decodes on the Triton path (fused_sigmoid_gating 74 us + DeltaLog materialize per GDN layer) instead of the CUDA MTP kernel (37 us) | ~48 x 40 us = 2 ms per mixed step x 400 = **~0.8 s, ~+0.8%** (T2 noted ~0.5%) | the CUDA MTP kernel can take the spec rows of a mixed batch (it is indexed per request) | DeltaLog materialize ordering in mixed steps | graph-trace kernel times already measured; build is moderate |
| - | Dropped after comparison: decode GEMM tiles/prefetch (at achievable floor, T2-010/T3-013), k=3 (T2-013), hostmem for 1.4 MB (T3-001/job 74), ReplaySSM (T3-007), lm_head kernels, PP/DP, fp16 accumulation | | | | |

Plan: E1 answers #1 and the ground truth for #2 in one job (~20 GPU-min). If #1 wins: bank it, check ABBA, official.
If the steplog shows small mixed steps at >= ~60 ms: build #2 (worker), starting with a prototype that measures how much of
the GDN core is graph-capturable on current metadata. Otherwise, drop #2 and look at #3.
