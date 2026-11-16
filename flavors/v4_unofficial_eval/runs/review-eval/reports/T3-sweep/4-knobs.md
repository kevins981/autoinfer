# T3 sweep, step 4: knob by knob

Reference: the champion as calibrated in `3-calibration.md`. Benchmark wall time ~118 s per start = **decode ~68%** (36.0 ms/step,
~2,440 steps) + **mixed prefill steps ~32%** (~121-143 ms/step, ~325 steps). Gains are given as ms/step and as % of the benchmark
(napkin, before acceptance or efficiency risk). "exact" = same math (bitwise or fp reordering); "draft-side" = only the drafter changes
(verification stays exact).

## Knob 1: Keep / Recompute (data nodes across steps)
| site | candidate | rules | napkin | verdict |
|---|---|---|---|---|
| GDN state S[r,l] (in-place, fp32) | **K1-a DeltaLog for spec decode**: write only S after row u=0 (always accepted) + keep the tiny per-row inputs (k,v,g,beta of u=1,2: ~2 x 24 x 256 floats = 50 KB/req/layer) ; next step replays the accepted 0-2 rows from S in SRAM before its own rows | valid (Rule 2: inputs are t-free once produced); exact if the replay does the same fp32 ops in the same order | napkin traffic per layer 4 states -> 2 states (+ <1%): -1.7 ms/step if all 3 slots really hit DRAM (=-4.7% decode, **-3.2% total**). But the kernel already runs below that napkin (calibration D2), so the real saving may be smaller | candidate; **first test: ncu dram__bytes of gdn_decode_post_conv_mtp** |
| GDN state | keep S in bf16 / fp16 | approx (state precision is part of the GDN math in the contract) | -1.2 ms | **dropped: contract** (GDN math fixed; fp32 state is the model's `mamba_ssm_dtype`) |
| KV cache k,v (16 layers) | keep (current) vs recompute | — | KV is 0.13 GB/step | keep is optimal (framework 02 §3) |
| logits / lm_head output | keep target logits of row u for the drafter | Rule 1: the drafter needs MTP hidden, not target logits | 0 | dead (no future reader) |
| prefix cache | keep (on, required) | no shared prefixes in ShareGPT | 0 | fixed by contract |
| draft tokens / MTP KV for rejected rows | already append-only | — | 0 | nothing to gain |

## Knob 2: Rewrite (identities, dedup, merges of nodes that share inputs)
| site | candidate | napkin | verdict |
|---|---|---|---|
| GDN `in_proj_ba` (bf16, N=48 per GPU) | **K2-a one small fused GEMV kernel** (or merge into a bf16 GEMM with the same input xn) instead of cuBLAS split-K 27 + splitKreduce (6.8 + 18.1 us) | 0.5 MB at 1 TB/s + launch ~3 us: -1.1 ms/step decode (-3.1% decode), also ~1 ms per mixed step: **-2.3% total**; exact (fp reordering in K) | **dropped** (review T3-1 + job 45: runs on a side stream overlapping qkvz; Triton 4.2 vs F.linear 5.8 us isolated) |
| `in_proj_ba` merged into the FP8 qkvz GEMM | would quantize a bf16-only layer | — | **dropped: contract** (not the checkpoint's FP8 scheme; ba is in modules_to_not_convert) |
| drafter lm_head (bf16 target head, 2x/step) | **K2-b fp8 copy** (`draft_lm_head_quantization=fp8`, ruling 1: allowed) | 2.71 -> ~1.4 ms: **-1.3 ms/step = -3.6% decode, -2.5% total**; acceptance may move slightly | candidate (flag only; must work with local argmax; not in the champion) |
| drafter lm_head | **K2-c reduced-vocab draft head** (top-V' tokens from non-benchmark text, ruling 2) | V'=32K fp8: 2.71 -> ~0.2 ms: -2.5 ms/step = **-4.6% total** gross; minus acceptance loss on tokens outside V' | candidate, needs a generic token-frequency source + acceptance test |
| residual add + RMSNorm (+ FP8 quant) after each all-reduce | **K2-d fuse into the HMAR kernel epilogue** (the AR already holds both partials of the full row; add residual, norm over d, write bf16 + the FP8 group quant for the next GEMM) | removes 128 add+norm (0.41 ms) + 128 quant launches (~0.3 ms) + extra passes: **-0.7 ms/step = -1.9% decode, -1.3% total**; exact (same per-row ops) | candidate (medium effort) |
| quant after SiLU-mul (66/step) | fuse act_and_mul + group quant | -0.2 ms/step, -0.4% total | small, bundle with K2-d |
| torch.compile (mode is NONE) | inductor fusions of elementwise/norm/quant | overlaps K2-d; launch floors are already graphed | low priority (startup time, fusions small at 48 rows) |
| attention output gate, qk-norm+rope | already fused kernels | ~0 | nothing |
| associativity (MLA-style absorption) | none: standard GQA attention with RoPE; GDN chunk form vs recurrent form already chosen by row count (prefill chunked, decode recurrent) | 0 | covered |

## Knob 3: Speculate (edges: next token, drafts)
| site | candidate | napkin | verdict |
|---|---|---|---|
| next-token edge, chain depth | **K3-a num_speculative_tokens 3** | acceptance per position 0.78/0.56 -> assume p3 ~0.40: tokens/step 2.34 -> ~2.74 (+17%). Cost: M 48 -> 64 rows (GEMMs still below the FP8 ridge ~82 rows: +~3% GEMM time), +1 draft step (1.36 ms bf16 head, 0.7 fp8, + 0.3 MTP), AR 46 -> ~55 us (+1.2 ms), GDN +1 slot (+0.8 ms?): step 36 -> ~40 ms (+11%): **net ~+5% decode, ~+3.5% total**; break-even p3 ~0.30 | candidate; better after K2-b (cheaper draft step) |
| tree speculation (top-2 at the first draft position) | M per request 3 -> 5; GDN state per tree branch (S has no j index: each branch needs its own state slot), MTP KV per branch | GDN rollback per branch makes trees costly here (framework 02 §4.4); first-position top-2 adds ~0.1-0.15 tokens/step | dropped for implementation cost (corrected, review T3-1: the framework's checkpoint + private-suffix form does give branches a shared state; building it in the GDN kernels and MTP is the cost) |
| draft quality | retrain/distill MTP head on generic data | unknown, + up to 10-20% tokens/step | dropped: GPU-heavy training beyond this team's budget |
| other edges (router, layer outputs) | no MoE; layer outputs are high-cardinality | — | fail Gate 2 (framework 04 §3) |

## Knob 4: Split / Merge (indices)
| index | candidate | napkin | verdict |
|---|---|---|---|
| m, h, hv, v across 2 GPUs (TP=2) | current | — | baseline |
| l across GPUs (PP=2) | 2 microbatches of 8 req | each stage holds half the layers unsharded = 12.17 GB (corrected; earlier text said 6.1): a 1-microbatch step leaves a GPU idle half the time, a 2-microbatch step reads each stage's 12.17 GB twice (24.4 ms of weight time per GPU) | **dropped**: shared-read rule; PP=2 with 2 microbatches costs 2x weight reads (+12 ms) to save 6.2 ms of AR. Breaks only if AR > weight time (not here) |
| r across GPUs (DP=2) | full weights per GPU | 24.3 GB FP8 + lm_head > 24 GB | **dropped: capacity** |
| asymmetric TP split (alpha != 0.5) | — | ranks do symmetric GPU work (T_other equal) | dropped (framework 05 §4.1: symmetric optimal when T_other equal) |
| d across ranks for the residual (sequence/row-parallel: reduce-scatter + all-gather) | — | same link bytes, 2 dependent exchanges instead of 1 per site: worse on a latency-bound link | dropped |
| **message index (rows x 5120) of a mixed-step all-reduce** | **K4-a chunked, pipelined host-staged all-reduce for large messages** (GPU0 writes chunk k to host while GPU1 reads chunk k-1; both directions at once) replacing NCCL RING_LL (363 us for 2.66 MB = 7.3 GB/s) | link floor ~110-130 us for 2.66 MB: -230 us x 129 = **-30 ms per mixed step = -23% of mixed, -7.3% total**. Cheapest first look: NCCL env (NCCL_PROTO=Simple/LL128, channels, buffsize) in a microbenchmark | **top candidate**; exact (fp32 sum of 2 bf16, rounded once: same as NCCL for 2 ranks) |
| **output index d of the GEMM feeding an all-reduce (W_o, W_out, W_down)** | **K4-b column-split GEMM + per-chunk AR, chunks overlapped** (chunk 1's AR on the link while chunk 2's GEMM reads its half of the weights: no weight is read twice) | per site: GEMM ~20-40 us, AR ~46 us; with 2 chunks hide ~min(GEMM/2, AR/2) - extra AR latency: ~15-20 us x 128 = **-2-2.5 ms/step = -6% decode, -4% total** | **dropped** after worker microbench (job 45/46, D): serial 97.3 us vs split 122.6 (latency-bound half-AR) / 97.0 (optimistic) |
| r split into 2 microbatches inside a decode step (DBO) | overlap AR(mb1) with GEMM(mb2) | shared-read rule: 48 rows are below the ridge (~82), splitting doubles weight reads (+12 ms) to hide <=6 ms | **dropped for decode** (framework 06 §3) |
| **r split into 2 microbatches inside a mixed step (DBO)** | **K4-c / K6-a** mixed steps are compute-bound (260-420 rows > ridge 82 per half), so the double weight read hides behind compute; overlap NCCL AR (47 ms) with the other microbatch's GEMMs (45 ms) | up to -35-40 ms per mixed step (**-9-10% total**) if overlap were perfect; competes with K4-a for the same 47 ms (after K4-a, ~17 ms left to hide: -3.5% more) | **dropped**: V2 ubatching needs DP>1 (gpu/ubatch_utils.py:598), hybrid models unsupported (config/vllm.py:3718), mamba asserts ubatch 0, HMAR disabled under ubatching (review T3-1) |
| i (prefill chunk size, max_num_batched_tokens 2048) | smaller chunks | prefill is compute-bound; chunking doesn't change FLOPs; TTFT/TPOT trade | unpriced (corrected: chunk size also changes weight passes, graph padding, repeated decode work and head/scheduling overhead) |

## Knob 5: Place (devices, memory tiers)
| node | candidate | napkin | verdict |
|---|---|---|---|
| **next GEMM's weights -> L2 during each all-reduce** | **K5-a L2 weight prefetch in the AR window**: HBM is idle while the HMAR kernel waits on PCIe (46 us x 135 = 6.2 ms/step); a side-stream kernel prefetches the next GEMM's first ~30-40 MB (L2 72 MB) so that GEMM reads part of its weights from L2 | per AR up to ~40 MB moved off the critical path: ideal -30-40 us x 128 = -4-5 ms; realistic half: **-2-2.5 ms/step = -4% total** | **built: T3-001, +3.7% on T2-007 (816.6 tok/s, job 49)**; microbench job 47: -10 us per AR+GEMM pair at 16 MiB |
| KV / GDN state offload to host | no capacity pressure (5 GiB KV budget is ample at c=16) | 0 | dropped |
| lm_head placement | split over v (done); the drafter reads the same split | — | covered by K2-b/K2-c |
| CPU work (scheduler, drafter metadata rebuild "eagerly" for FlashInfer) | GPU idle 0.8 ms/decode step, 9.5 ms/mixed step | -0.8 ms decode (2.2%) + up to -9 ms mixed (2.4%) if fully hidden | candidate to investigate (graph-level nsys first: node-level traces inflate gaps) |

## Knob 6: Window (merging / overlapping independent nodes)
| pair | candidate | napkin | verdict |
|---|---|---|---|
| AR (link) || next-layer weight load (HBM) | = K5-a | | |
| AR (link) || second half of the producing GEMM | = K4-b | | |
| mixed step: AR(mb1) || GEMM(mb2) | = K4-c | | |
| target lm_head (48 rows) || drafter lm_head step 1 (16 rows) | merge their weight reads | dependency: draft step 1 needs the target's sampled token -> must be in a later window | **dropped (Window dependency rule)** |
| draft step 1 || draft step 2 (shared lm_head and MTP weights) | merge = parallel drafting | serial edge (draft token); breaking it needs a drafter trained for parallel drafting | dropped (needs training) |
| GEMM efficiency (implementation layer, not a knob) | **K7-a better decode GEMM kernels** (Triton configs per shape at M=48, split-K/stream-K for gate_up 0.69 TB/s) | GEMMs 16.9 ms at 0.72 of DRAM peak; to 0.85: -2.6 ms/step (**-4.9% total**) | candidate (the framework has no knob for kernel efficiency; this is the "Implementation" layer of 01 §3) |
| per-step small kernels (264 quant, 132 norm, 66 act, conv, misc ~1.5 ms) | fuse (K2-d) | | |
| ncclSendRecv 454 us per decode step | find what it is (batch-sharded sampling exchange?) and whether it is a wait (rank skew) or work | up to -0.45 ms/step (-0.9% total) | **corrected: real transport** — batch-sharded sampling's vocab-shard all-to-all of target logits (~6 MB/rank, model_runner.py:1719). New candidate K2-e: exact greedy fast path (local max/idx per shard + tiny gather + greedy verify), ~-0.45 ms/decode step, ~+1-1.5% |

## Summary of surviving candidates (napkin, % of benchmark time)
| id | candidate | napkin gain | exact? | effort |
|---|---|---|---|---|
| K4-a | pipelined host-staged AR for mixed-step (large) all-reduces / NCCL tuning | ~7% | exact | env test cheap; kernel medium |
| K4-c | DBO for mixed steps | up to 9% alone, ~3.5% after K4-a | exact | high |
| K7-a | decode GEMM efficiency | ~5% | exact | medium |
| K2-c | reduced-vocab draft head | ~4.6% gross | draft-side | medium |
| K4-b | column-split GEMM + overlapped AR | ~4% | exact | medium-high |
| K5-a | L2 weight prefetch during AR | ~4% | exact | medium, risky |
| K3-a | 3 draft tokens | ~3.5% | exact | flag (+memory/128K check) |
| K1-a | GDN DeltaLog state writes | <=3.2% (napkin doubtful) | exact | high |
| K2-b | fp8 drafter lm_head | ~2.5% | draft-side | flag |
| K2-a | fused ba GEMV | ~2.3% | exact | low |
| K5-cpu | CPU gaps | ~2-4% | exact | unknown |
| K2-d | AR + residual + norm + quant fusion | ~1.3% | exact | medium |
| SendRecv | per-step exchange | ~0.9% | exact | low |
