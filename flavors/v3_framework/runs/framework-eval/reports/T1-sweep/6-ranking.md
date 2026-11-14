# T1 sweep, step 6: ranked candidates

Gains are napkin, vs BASE's measured time (decode step 38.5 ms = 72.5% of time, mixed step 148 ms =
27.5%), overall = 0.725·ΔD/38.5 + 0.275·ΔP/148. Stacking is not additive (several attack the same ms).

| # | candidate (sites: 4-knobs.md) | napkin overall | wins if | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | **E3: zero-copy all-reduce through mapped pinned host memory** (both GPUs write own partial, read peer's, same-order add → bitwise identical on both ranks) | **~+10%** (D -3.8 ms, P -18 ms) | GPU stores to / loads from host memory reach >15 GB/s per direction concurrently, flag latency <5 us | GPU reads of host memory (non-posted PCIe reads) are much slower than writes; polling host flags adds latency | microbench vs NCCL at n=42/375 (**running**, worker w1) |
| 2 | **Cheaper drafter + deeper chain**: FP8 draft lm_head (flag), local argmax for drafts (flag), then num_speculative_tokens 3-4; later a reduced draft vocab (non-benchmark token frequencies) | **~+5% (drafter) then +4-8% (depth)** | the draft step cost falls from ~2.25 ms to <1 ms and conditional acceptance at depth 3 stays >~0.65 | per-position acceptance at depth 3/4 (unknown; I assumed 0.72 conditional) and that the extra 16 verify rows are cheap (AR grows with rows: +1.5 ms/step on NCCL) | serve-flag A/Bs (no code): B = fp8 draft head + local argmax; then k=3 on top |
| 3 | **W1: overlap AR with adjacent GEMMs** (split row-parallel output along d; next GEMM on arrived d-chunks, RMSNorm scale commuted) | ~+9% alone; less after #1 | AR kernel and Marlin can run concurrently without slowing Marlin more than the AR time hidden | SM / L2 contention (framework limitation #3), and capture of multi-stream overlap inside CUDA graphs | microbench: Marlin gate_up/down at M=42 alone vs concurrent with an AR on another stream |
| 4 | **R5: W8A8 FP8 GEMMs for mixed (prefill) steps** (per-token-group-128 dynamic activations, allowed) | ~+6.5% (P -35 ms) | a block-scaled FP8 GEMM on sm_89 reaches >~180 TFLOP/s at M≈300-400 | kernel availability/speed on sm_89 (cutlass blockwise is sm90+; Triton path unknown) + GSM8K A/B within 2 pts | microbench Triton `w8a8_block_fp8_matmul` vs Marlin at M=375 on the 4 real shapes |
| 5 | **E1: Marlin decode efficiency** (0.66 in-situ vs 0.8 isolated BW) | ~+5.8% (D -3 ms) | the in-situ gap is a tunable (config/launch) not physics | that ncu's isolated number is reachable in-situ | microbench Marlin at M=42 isolated, and right after an NCCL AR |
| 6 | R1 target-side local argmax (greedy rows; generic fallback) | ~+2% (D -0.84 ms, P ~-1 ms) | all requests greedy without logprobs (true for the benchmark, must stay general) | sampler/rejection path needs full target logits for other features | code read + one A/B |
| 7 | Draft attention metadata in place / idle removal | ~+2.5% | idle 1.3 ms/step is CPU-side draft prep | it's elsewhere (sampling, scheduler) | TRACE=graph profile of the champion |
| 8 | Mixed-step two-microbatch overlap (DualPipe-style) | up to +7%, shrinks after #1/#3 | mixed step stays compute-bound and the AR is still large | eager-mode complexity | after #1 |
| 9 | ReplaySSM for GDN state, GDN kernel efficiency | ~+1-2% each | | | later |
Dropped (with reasons in 4-knobs.md): PP / asymmetric PP, sequence parallel, lm_head replication,
lossy state, tree speculation (for now), draft-in-verify bubble, NCCL env tuning (measured), FlashInfer
PCIe-IPC (impossible without peer access).

Plan: #2's flag tests now (cheap, no code), #1 in parallel by the worker, then #3/#4/#5 microbenches.
