# T1 sweep, step 4: the sweep, knob by knob

Reference times (3-calibration.md): decode step D = 38.5 ms (72.5% of benchmark time), mixed step
P = 148 ms (27.5%). Overall gain ≈ 0.725·ΔD/38.5 + 0.275·ΔP/148. Gains below are napkin with the
calibrated efficiency factors. Legend: **keep** = goes to ranking; **drop** = pruned, with reason.

## Knob 1: Keep / Recompute (data nodes across t)
| site | candidate | napkin | verdict |
|---|---|---|---|
| `S[r,l]` GDN state (fp32, in-place) | Row A (now): read+write 1.57 MB/req/layer every step | 2.96 ms/step | baseline |
| | Row B ReplaySSM: keep `S[t-L]` + k,v inputs, write S every L steps | write half of GDN traffic ~ -1.0 ms D, but replay compute of ≤L·2.3 tokens and rollback logic in the MTP kernel | **keep (low rank)**: ~1.9% overall, high effort |
| | lower-precision S (bf16) | -1.5 ms | **drop**: lossy (contract: same math, no approximation) |
| `kv[r,j]` | fp8 now | fixed by contract | drop |
| prefix caching | generic default (on, mamba align) | ShareGPT prompts share almost nothing except template | drop: no gain, and tuning it to the benchmark is forbidden |
| `logits`, `q`, `s[t,j]` | keep | Rule 1 / Rule 2 fail | drop |
| recompute instead of keep (e.g. drop KV, recompute from x) | | everything is BW-bound at decode; recompute adds FLOPs and bytes | drop (dominated, 02 §3 table) |

## Knob 2: Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| `argmax_v` after vocab-parallel lm_head (target + 2 draft steps) | Distributivity of the (max,idx) reduction over the v split: local argmax per shard, exchange 8 B/row, pick (max, lowest idx) | removes 10.4 + 2x3.5 MB all-gathers: -1.43 ms D (0.84 + 0.59), ~-1.4 ms P | **keep**: ~2.9% overall; exact for greedy rows (ties: lowest index both sides). Needs a generic fallback when any request needs logprobs / non-greedy sampling |
| RMSNorm per-row scale before column-parallel GEMM | commute `rsqrt(mean x²)` past Σ_d | 0 alone; **enables** chunked AR→GEMM pipelining (Window W1) | keep as an enabler |
| draft lm_head (1.27 GB bf16, read 2x per step) | (a) FP8 draft lm_head (ruling 1) | -2 x 0.68 = -1.37 ms D | **keep** |
| | (b) reduced draft vocab (ruling 2; ids from non-benchmark text), e.g. 32K rows | 2 x (1.37 - 0.18) = -2.4 ms D, plus smaller gathers -0.5 ms; acceptance loss when target's token outside subset (~1% of tokens) | **keep**: ~5.5% overall; combine with (a) |
| body GEMMs in mixed steps (compute-bound at 375 rows) | W8A8 FP8 per-token-group-128 (allowed) | 81 ms → ~45 ms at 0.6 of 330 TF: -35 ms P | **keep**: ~6.5% overall. Riskiest: a fast block-scaled FP8 GEMM for sm_89 (cutlass blockwise is sm90+; Triton `w8a8_block_fp8_matmul` speed on 4090 unknown) + the GSM8K A/B |
| body GEMMs in decode (BW-bound) | W8A8 | same bytes; only wins if the kernel streams better than Marlin (0.66) | fold into E1 |
| in_proj_ba (bf16, separate tiny GEMM + splitK reduce) | merge into in_proj qkvz launch (Dedup of the shared input n1) | 48 x ~8 us = -0.4 ms D | keep (low rank, ~0.8%) — but weights are bf16 vs fp8: needs a mixed kernel; probably just a better small-GEMM kernel |
| MLA-style absorption, LoRA factoring | | no such structure (no latent KV, no adapters) | drop: not applicable |

## Knob 3: Speculate (exact only)
Gate 2 (cardinality): only the token edge (and nothing else: no MoE router, no block-sparse index) is
low-cardinality. Every activation edge (y[l], AR outputs) is high-cardinality → exact speculation p≈0.
| site | candidate | napkin | verdict |
|---|---|---|---|
| token edge t→t+1 | chain depth k=2 (now): acceptance 2.345 (pos 0.78/0.56) | | baseline |
| | k=3 / k=4, current drafter | conditional accept ~0.72 → acc 2.75 / 3.04. Cost per extra depth: +16 verify rows (Marlin ~+0.9 ms, AR +16 rows x 0.72 us x 135 = +1.55 ms, GDN +0.3 ms) + 1 draft step (2.25 ms now) ≈ +5 ms → k=3: 44 tok/43.5 ms vs 37.5/38.5: **+3.8%** | keep, but much better after R3 |
| | k=3/4 **with cheap drafter** (R3 a+b: draft step ~0.8 ms) and fast AR (E3) | per depth ≈ +2.5 ms → k=3 ≈ +9%, k=4 ≈ +11% over the same base | **keep** (stacked) |
| | tree drafts (top-2 at depth 1) | GDN state must fork per tree node (02 §4.4 trap): needs a tree-aware GDN verify kernel | drop for now: big kernel work, gain ~+5% only after chain depth is exhausted |
| layer edge y[l]→y[l+1], AR outputs | | high cardinality, weights carry l | drop (exact p=0) |
| draft-in-verify-bubble (06 §4.2): drafts of cohort R1 during R2's link time | split r doubles weight reads (Shared-Read Rule) at N=48 ≪ ridge | loses ~+18 ms of weight reads to hide ~8 ms of link | drop (dominated) |

## Knobs 4 & 5: Split / Merge and Place
| site | candidate | napkin | verdict |
|---|---|---|---|
| layers | PP=2 (32+32 layers, 1 crossing per forward instead of 128 ARs) | decode: each GPU streams 12.2 GB serially (no TP halving per step): ~37 ms weights + lm_head 2.5 GB on one GPU → ~48 ms vs 38.5; 2 microbatches re-read weights (Shared-Read Rule) → worse | **drop for decode** (dominated: TP halves bytes per GPU per step, the AR costs less than 12 ms) |
| | PP in mixed steps only | impossible to switch per step without a 2nd copy of weights (capacity 24 GB) | drop |
| | asymmetric 33/31 split (05 §4.1) | only matters if PP | drop with PP |
| `m`, `h`, `hv` | TP=2 (now): 2 ARs/layer | | baseline |
| | sequence parallel (reduce-scatter + all-gather instead of AR) | same link bytes, 2x the latency terms | drop |
| `d` of row-parallel output | split d into chunks + AR per chunk (pipelined) | see Window W1 | keep (as W1) |
| `v` (lm_head) | replicate lm_head on both GPUs | each GPU reads 2.5 GB instead of 1.27 per lm_head call, and -1.27 GB of KV budget | drop |
| combine node implementation (Place the AR buffers in pinned host memory, zero-copy kernel) | E3 below | | keep |
| KV / state to host tier | capacity not binding (5 GiB KV fits 16 req; 128K fits) | | drop |

## Knob 6: Window
| site | candidate | napkin | verdict |
|---|---|---|---|
| AR (link) || next GEMM (HBM) inside a layer | **W1**: split the row-parallel GEMM output along d into c chunks, AR chunk k while computing chunk k+1; start the next column-parallel GEMM on arrived d-chunks (RMSNorm scale commuted, R2). Splits on d don't re-read weights (different columns) | hides up to min(AR, adjacent GEMMs): at decode ~50% of 7.7 ms → -3.8 ms D; mixed ~-20 ms P → **~9-11% overall** | **keep** (high effort: concurrent NCCL/custom AR + Marlin inside graphs; framework limitation #3: SM sharing) |
| mixed steps: two row-microbatches staggered (DualPipe/NanoFlow, 06 §3) | mixed step is compute-bound (375 rows ≫ ridge ~80): Shared-Read Rule cost is hidden → AR of half 1 overlaps GEMMs of half 2 | -up to 39 ms P → ~up to 7% overall | keep (medium; only mixed steps; eager path) |
| CPU work between graphs | async scheduling / in-place draft metadata (now rebuilt eagerly, server log) | idle 1.3 ms D (3.4%), 6.4 ms P | **keep**: ~2.5-3% overall, cheap test (flags / attention backend that supports in-place update) |
| mixed steps eager launches | full CUDA graph for mixed batches | ~5 ms of P | keep (low, ~1%) |
| target lm_head with draft lm_head in one window | serial dependency (draft input needs the accepted token) | | drop (Window Dependency Rule) |
| phase-shifted KV scan (06 §4.1) | attention KV scans are 0.5 ms total | | drop (too small) |

## Outside the framework's knobs: node implementation efficiency (calibration gaps)
The framework prices nodes at hardware rates; here the biggest gaps are kernels. I list them as sites
because they are where the measured time is.
| site | candidate | napkin | verdict |
|---|---|---|---|
| **E3** combine node (AR) | custom zero-copy AR through mapped pinned host memory (both directions concurrently, no P2P needed) | 58 → ~25-30 us at 42 rows: -3.8 ms D; 294 → ~160 us at 375 rows: -18 ms P → **~10.5% overall** | **keep (top)**. Worker microbench in flight. NCCL tuning measured: nothing beats default (job 1); FlashInfer PCIe-IPC backend needs peer access → impossible here (job 1) |
| **E1** Marlin at M=42-64 | tune Marlin configs / an FP8 GEMM that streams at 0.8 BW (ncu shows 0.8 isolated vs 0.66 in-situ) | -3 ms D → **~5.8% overall** | keep. Cheapest test: microbench Marlin real shapes M∈{42,48,64} isolated vs after an NCCL call |
| E2 GDN decode kernel 0.71 BW | better streaming of fp32 state | -0.6 ms D (1.1%) | keep (low) |
| E4 GDN decode rows inside mixed steps use `fused_sigmoid_gating_delta_rule_update` (83 us/layer vs 62 us) | use the MTP decode kernel there too | -1 ms P (0.2%) | drop (small) |
