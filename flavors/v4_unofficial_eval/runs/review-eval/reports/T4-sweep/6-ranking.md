# T4 sweep, step 6: ranking (against the champion T3-003, 826.1 tok/s)

Gain rule (3-calibration.md C): x ms saved per decode step -> x/32.7 x 0.62; y ms per mixed step -> y/140 x 0.38.

| # | candidate | napkin gain | wins if | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | **K6-c: remove the fixed excess of mixed steps** (small-P mixed steps cost 72-122 ms vs 32.7 for decode; ~40-85 ms per mixed step not explained by compute) | if half of ~50 ms goes: 25/140 x 0.38 = **+7%**; all of it up to +15% | the excess is overhead (eager GDN prefill core and attention between graph pieces, CPU metadata for prefill, eager draft in mixed steps, peer skew), not required work | that the excess is avoidable with code we can change in the time left | CPU-only: per-phase breakdown of small mixed steps vs decode (worker task 2, running); then one quick bench of the specific fix |
| 2 | **K6-a: chunked prefill into decode steps** (`--max-num-batched-tokens` ~96-160 so prompt rows ride decode steps) | +14% napkin if a decode+~50-row step costs ~37 ms; **loses** at today's mixed-step cost | #1 fixed (a small mixed step ~ decode step + compute of its rows) | that a small-P step can approach decode cost (FULL graph vs piecewise); TTFT p99 stays < 1609 ms | quick bench of the flag on T3-003 (1 start, ~5 GPU-min) after #1's analysis; expected negative before #1 |
| 3 | **K3-a: 3 draft tokens** on T3-003 (FP8 draft head makes the 3rd draft step ~0.8 ms) | tokens/step 2.33 -> ~2.7 (p3 est. ~0.40); step +~3.5 ms -> **+4-5%** | p3 >= ~0.33 and memory fits (KV >= ~4 GiB, graphs for M=64) | acceptance at position 3 and memory headroom (T3-003 already trimmed KV to 4.4 GiB) | 1 quick-bench start of the flag with startup memory lines (128K test if it fits); T1-009's 764 on older code is a screen only |
| 4 | **K1-a: GDN DeltaLog** (write only row-0 state; replay accepted rows next step from inputs kept in slot 1) | 72.7 us/layer -> ~45-60: -0.6 to -1.4 ms/decode step -> **+1.1-2.6%**; also makes #3 cheaper (no 4th state write) | the in-server kernel is state-write-bound (T3: cold 120/89/60 us for 3/2/1 writes; in-server 70 us) | that writes, not reads/L2, dominate the in-server kernel; every reader of spec slots (mixed-step Triton path, prefix-cache copies) adapted | microbench of the existing kernel with destination slots set to 0 (1 write) vs 3 in a warm, server-like loop (~1 GPU-min) |
| 5 | **K45-c: host-mapped AR for mixed-step sizes** (hmar beyond 96 rows) | bare NCCL ~205 us at 2.66 MB vs hmar's per-row cost about the same (T1): **~0-2.7%** | only if hmar's per-row cost beats NCCL at 1-4 MB and skew is gone | T1 found equal per-row costs | already answered mostly negative by T1's microbench: dropped unless #1 shows the AR itself (not skew) is the excess |
| 6 | K2-b reduced-vocab draft head | <0 after FP8 head (acceptance -4.6% in T1-002) | — | — | dropped |

Groups dropped (see 4-knobs.md and 5-compare.md): pipelined/ordered hmar (T1 tested: slower), DBO (blocked in
this vLLM), PP/DP, trees, approximate variants (contract).

Plan: #1 analysis first (CPU, running). Then GPU: one job with quick benches of #3 (k=3) and #2 (token
budget) on T3-003 to screen them cheaply, while #1's fix is designed. #4 only if a cheap microbench shows
the writes matter in a warm loop.

## Addendum (same session, after worker task 2: agents/leader-4.w1/mixed.md)
#1's cause, observed on two ~60-row mixed steps (77 / 116 ms) vs a decode step (34 ms), node trace, both GPUs:
identical kernels and GPU durations in both mixed steps; the main forward is 68 graph pieces + ~1730 eager
launches (all GDN prefill kernels ~25-30 per GDN layer, full attention, ~1000 tiny copy/gather ops); no GPU
gap > 200 us, but thousands of 10-200 us gaps where the next launch started after the GPU went idle; the CPU
thread issues ~3130 runtime calls per mixed step (159 in decode). GPU1's thread is the slower one, so GPU0
waits in NCCL (~19-42 ms of its AR time is peer wait, transfer ~14 ms). The 77 vs 116 ms split is a run-phase
effect (per-launch CPU cost 1.4x higher early in the run: host contention).
=> mixed steps are **CPU-launch-bound**: ~25-65 ms of their ~75-120 ms. The GPU floor of a ~60-row mixed step is
~52 ms.
Refined #1 (building, worker task 3): run short prompt chunks' GDN prefill as ONE recurrent Triton kernel per
layer (03 §4.2 dual form: token-by-token form, exact identity, fp reordering only) instead of the FLA chunk
pipeline (~12 Triton launches + ~6 torch ops per GDN layer). Napkin: -~550 Triton launches and -~300 torch ops
per mixed step at ~25-35 us CPU each -> -15..25 ms CPU per mixed step, minus the extra GPU time of the
sequential form (to be measured; must stay below the CPU saved) -> ~+4-7%. Riskiest assumption: recurrent GPU
time at T ~ 260 rows per layer is small (< ~150 us/layer). Cheapest test: microbench (correctness, GPU time,
CPU time per call vs T).

## Review T4-1 (reviews/T4-1.md): my answers
1. Decode shares are node-sum/graph-period hybrids; ba overlaps qkvz; small kernels ~3 us not 5-6: ADOPTED (wording; no rank change).
   Mixed-step peer wait vs GPU1 idle are the same delay, not additive: ADOPTED. GPU-work floor of a small mixed step ~50-52 ms:
   ADOPTED; launch fixes alone cannot make a 37 ms mixed step. "Host contention" is unproven: ADOPTED (cause of the 78/118 split open).
2. 62/38 is a constructed scenario, not a measurement (profile: 309 input tok/req, 10 decode reqs/step on average): ADOPTED;
   percentages in this file are sensitivity numbers (use 1/(1-s)-1 for finite savings).
3. With a prefill present, the decoding requests also lose the fused CUDA GDN path (general core: spec/non-spec gather/scatter,
   Triton spec recurrence, separate gated norm, contiguous copies): ADOPTED as part of #1 (both halves of the GDN mixed path
   are launch-count targets). Recurrent helper stores every token's state -> dedicated final-state kernel: already the plan.
4. #2: sub-block chunks are allowed in align mode; HMAR/L2-prefetch discontinuity at 96 padded rows; small chunks repeat per-step
   costs: ADOPTED. #2 demoted to a conditional screen after #1; "fewer, larger prefill-bearing steps" added as a row (latency trade).
5. k=3 adds a GDN speculative state block (~1.1 GiB/GPU at 16 req) -> capacity is the binding problem; reprice ~+3.5% before memory
   effects, break-even p3 ~0.25: ADOPTED. #3 stays below #1 and needs a capacity plan first.
6. DeltaLog contract includes prefix-cache temporal copies (accepted-1 offsets) and replay cost ~1.33 rows/step: ADOPTED. Stays #4.
Softened: reduced-vocab family "low priority" (not established negative); draft-during-collective overlap not ruled out (unpriced).
Revised order: #1 mixed-path GDN launch work (recurrent prefill + keep fused spec decode in mixed batches), #4 DeltaLog feasibility,
#3 memory-feasible k=3, #2 small-budget prefill (conditional).
