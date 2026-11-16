# T8 sweep 6: ranking (vs champion T5-001, 895.4 tok/s; 07:12 UTC, ~60 min before the run ends)

| # | candidate | napkin gain | wins if | riskiest assumption | cheapest test | status |
|---|---|---|---|---|---|---|
| 1 | Mixed-step excess (non-GEMM part: eager GDN prefill core, DL materialize ~1.5 ms, piecewise graph breaks, CPU metadata) | mixed steps carry ~40+ ms each beyond ~29 ms of GEMM compute; halving that excess on ~300-400 mixed steps: -6..-8 s of 101 s, **+6-8%** | the excess is removable overhead, not rank skew or AR waits | the excess decomposition (no per-op profile of a mixed step on T5-001 by T8) | nsys node trace of 32 prompts on T5-001, mixed steps only (~3 GPU-min) | untried as a whole; T4 named it (K6-c) but did not build it; too large to build before 08:15 |
| 2 | Multi-step GDN state log (write S every 2nd decode step) | **+2-2.5%** | state writes still cost ~1 ms/step after DL plus the spill into next kernels | replay of 2 logs stays within the kernel's shared-memory and latency budget | ncu of the DL kernel with the write skipped (NOSTORE env exists: VLLM_T5_DL_NOSTORE) | untried; large build |
| 3 | AR + residual + RMSNorm fusion | +1% | the norm passes are not already fused into hmar | — | — | T4 had it; low |
| — | k=3, larger L2 prefetch, reduced-vocab draft head | ~0 | — | — | — | tested negative or neutral (T7-001, T1-009, T5-002, T1-002) |
| — | fp16 accumulation in W8A8 | +2-3% | — | — | — | **not allowed** (ruling); bank/T8-001 |

None of the top two can be built, reviewed and measured with the harness before 08:15. The cheapest useful step is #1's profile.

## Evidence for #1 (07:14, no GPU spent: the champion's own nsys node profile, job 118, agents/leader-5.w1/prof_node_dl/steps.txt)
Observed: 52 mixed steps (median 215 prefill + 45 decode tokens): period 119.5 ms, GPU busy 90.1, GPU idle in window 27.3 ms;
NCCL all-reduce 37.2 ms (~285 us per AR at ~2.6 MB vs a ~100 us PCIe floor); uncategorized (Triton GEMMs + prefetch) 39.4; GDN 7.3.
Decode steps: 491, period 25.4 ms (node trace inflates spans).
Interpretation: at 240-275 TFLOP/s (job 124), 260 rows need ~25 ms of GEMM, so ~65 ms of each mixed step is AR waiting plus GPU idle
(CPU-side eager work or rank skew). That is the largest lever left (profile slowdown included, plausibly +5-10% end to end). T6's
transport-only attempts (HMAR768, row split) did not move it, which suggests idle/skew rather than the AR transport. The next test
is TRACE=graph on a mixed step to see what the CPU does during the 27 ms idle.

## Update (job 125, bank/T8-002): TRACE=graph profile
A mixed step has a ~110 ms period floor independent of prefill size up to ~450 tokens; with 60-130 prefill tokens the GPU is busy
only ~50 ms (decode step 22.9 ms). So #1's excess is a fixed per-mixed-step host/sync cost: +15-25% napkin if it holds at harness
speed. The top lever for the next team.
