# T5 sweep, step 4: the sweep, knob by knob

Reference: HEAD measured in job 202 (200 prompts, 54.1 s): decode steps 31.8 s (pure-16 step 27.6 ms),
mixed steps 22.4 s (183 x ~100 ms). Conversion used below: **1 ms saved per decode step ≈ 1.2 s
≈ 2.3% of the run**; **1 ms saved per mixed step ≈ 0.18 s ≈ 0.34%**. Gains are napkin, against
these measured times, using the calibration factors of 3-calibration.md.

## 0. A resource the framework lacks: host CPU time
The framework's window cost is max(FLOPs/F, HBM/BW, Link/Link_BW). On this machine the largest
single gap (41% of the run is mixed steps costing ~70 ms more than a decode step, flat in prefill
size) is **CPU time** to issue ~1,800 launches plus a host<->device sync that serialises CPU
preparation behind the previous step. I add a fourth resource per window, `CPU_time(W)` (launches x
~20-40 us, python per-layer metadata), and treat "async scheduling" as a Window decision across
CPU and GPU (step t+1's CPU work in the same window as step t's GPU work), and "CUDA graph capture"
as a Merge of many launches into one (a CPU-side analogue of the Shared-Read rule: one launch
cost for many nodes). Speculating `seq_lens` (an upper bound + device fix-up) is the framework's
Speculate knob on the edge `step t accepted count -> step t+1 metadata`; HEAD already applies it to
pure decode steps only.

## 1. Keep / Recompute
| site | candidate | napkin | verdict |
|---|---|---|---|
| GDN state S (t-free, in-place) | lazy commit every 2 steps (HEAD) -> every 3-4 steps (longer DeltaLog) | state traffic 1.8 -> ~1.5 GB: -0.3 ms/decode (~0.7%) | small; kernel page layout coded for 2 pending steps; **low priority** |
| GDN state dtype | bf16 state | -0.9 ms | **illegal** (changes the math; state dtype fp32 is part of the model) |
| KV cache | fp8 already, no capacity pressure | 0 | prune |
| conv state | in-place, ~0.1 MB/(r,l) | ~0 | prune |
| prefix states (prefix caching) | on by ruling; benchmark has no shared prefixes beyond the template | 0 (and tuning to it is forbidden) | prune |
| logits | not kept (local argmax) | 0 | done in HEAD |

## 2. Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| in_proj_ba (reads the same x̂ as in_proj_qkvz; Dedup of the input read) | a proper small-N kernel (or fuse into the qkvz kernel's epilogue/second output) instead of cuBLAS Kernel2+splitKreduce (18 us/layer for 0.5 MB) | -0.75 ms/decode (**~1.7%**) + same in mixed steps | **keep: cheap, exact** |
| RMSNorm before a column-parallel GEMM: x̂ = x·w/rms -> GEMM(x·w)·(1/rms) (Distributivity: pull the per-row scalar out of Σ_d) | enables starting the next GEMM on d-chunks as all-reduce chunks arrive (see §6 pipeline) | enabler | keep as part of §6 |
| lm_head argmax | already vocab-parallel local argmax | 0 | done |
| attention output gate, qk-norm, rope | fused already | ~0 | prune |

## 3. Speculate
| edge | candidate | napkin | verdict |
|---|---|---|---|
| serial token edge (MTP chain, 2 drafts) | 3 drafts: tokens/step 2.34 -> ~2.7 (if p3 ≈ 0.68) = +16%; step cost +~14% (AR bytes +33%, 1 more draft step, GDN replay coded for 2 only) | ~+2% now; ~+5% if AR is cheap | **conditional** on §6 AR work; re-price after |
| serial token edge, tree | tree drafts need per-branch GDN states (in-place S: framework 02 §4.4) | large state copies per branch | prune (cost of state branching ≥ the gain at p1 ≈ 0.78) |
| `accepted count -> next step metadata` (CPU) | seq_lens upper bound + device fix-up for **mixed** steps (HEAD does it for decode only) | removes the blocking sync: lets CPU prep (~10-25 ms) overlap the previous GPU step | **keep** (part of the mixed-step family) |
| router / expert edges | dense model | - | n/a |

## 4-5. Split / Merge and Place
| index / node | candidate | napkin | verdict |
|---|---|---|---|
| layers l (PP=2) | 32 layers per GPU, 2 micro-batches of 8 | each GPU reads its 12.2 GB per micro-batch: decode ~31 ms vs TP 27.6 (and ~22 after §6); removes 1.3 MB/token of AR for prefill | **prune** for decode (weight reads x2 at M ≪ ridge; decode is 59% of time). Hybrid (PP for prefill only) needs 2 weight copies: capacity impossible |
| requests r (DP=2) | full model per GPU | 24.3 GB FP8 + lm_head > 24 GB | **impossible** (capacity) |
| α ≠ 0.5 splits | GPUs symmetric, lm_head vocab-split evenly | 0 | prune (framework 05 §4.1: T_other equal) |
| drafter placement | MTP layer + draft head on one GPU only (no TP: removes 4 ARs + gathers) | GPU0 reads +0.43 GB per draft step (+0.9 ms) vs -0.3 ms comm | prune |
| d of row-parallel GEMM output (down, out_proj, o_proj) | split N into chunks so all-reduce of chunk k overlaps GEMM of chunk k+1 (no extra weight bytes: each chunk reads its own weight columns) | see §6 | keep |
| r in decode (dual batch) | R1 GEMM ‖ R2 all-reduce | doubles weight reads at M=48 ≪ ridge 80: +13 ms vs -5 ms | **prune** (framework 06 §3 shared-read rule) |
| r/i in prefill rows (dual batch, compute-bound) | chunked GEMM + AR overlap | HEAD has it, eager only, M ≥ 192 | partially built; re-price with §6 |
| KV / state placement in host memory | no capacity pressure | 0 | prune |

## 6. Window (incl. the CPU resource of §0)
| site | candidate | napkin | verdict |
|---|---|---|---|
| **all-reduce internals** (write own slot -> flag -> read peer slot: Link up and Link down are different resources) | **pipelined host all-reduce**: per-chunk flags so reading chunk c overlaps writing chunk c+1; more blocks | 42 -> ~24 us per AR: **-2.4 ms/decode (~5.5%)**; mixed steps: lift the 1.25 MB cap and replace NCCL LL (≈10 GB/s) for 1.25-16 MB: 2.5x faster AR, ~-10-20 ms per large mixed step (~1-2%) | **keep: high gain, exact (same bf16 sum), contained kernel change** |
| row-parallel GEMM + all-reduce (HBM vs Link) | fuse/overlap: GEMM N-chunks, AR per chunk (or in the GEMM epilogue) | remaining ~24 us AR mostly hidden under 37 us GEMMs: further -1.5 to -2.5 ms/decode (~4-5%) | keep, after the pipelined AR (riskier: SM sharing, ordering) |
| AR -> norm -> next GEMM pipeline over d-chunks (with §2 rewrite) | start gate_up / qkvz on arrived d-chunks | another ~1 ms | defer (complex) |
| **mixed step: CPU launches** (~1,800 per step, piecewise graphs + eager GDN/attention prefill) | (a) capture the mixed step in a full CUDA graph (GDN chunked prefill with padded/device-side metadata); (b) cut per-layer python/launch count (metadata once per step, fused small ops) | ~70 ms -> ~10-20 ms extra per mixed step: **-50-60 ms x 183 = 9-11 s (~17-20%)** for (a); ~half for (b) | **keep: the largest gap**; riskiest to build |
| mixed step: CPU/GPU overlap | remove the blocking `seq_lens.cpu()` in mixed batches (speculated upper bound + device fix-up, §3) | -10-25 ms per mixed step (2-4%) | keep (cheap if the FlashInfer prefill plan tolerates upper bounds) |
| mixed step count (Merge across r of prefills) | schedule waiting prefills together (fewer mixed steps; scheduler interval) | saves ~35 ms per merged request minus ~9 ms of idle slot: ~5-10% | keep (value must be chosen on non-benchmark prompts: manager ruling); its value shrinks if mixed steps get cheap |
| decode W8A16 GEMMs (implementation) | 79% -> ~90% of spec BW (one M-block of 48 rows, split-K for occupancy, cp.async pipelining) | -1.8 ms/decode (~4%) | keep (medium risk: earlier tuning may have saturated the Triton design) |
| W8A8 in mixed steps | ~560 GB/s effective at M ~ 100 | -5 ms per mixed step (~1%) | low priority |
| lm_head cuBLAS (800 GB/s) | better GEMV-like kernel / W8A16-style bf16 kernel at 930 GB/s | -0.2 ms (~0.5%) | low |
| small kernels (norm, act, conv, gathers: ~1.3 ms vs ~0.3 napkin) | fuse residual+norm+quant, act into GEMM epilogue | -0.5-0.8 ms (~1.5%) | low-medium |

## 7. Approximate rows (listed, all illegal by contract)
sparse / windowed attention, early exit, quantised all-reduce, lower-precision target lm_head,
bf16 GDN state, approximate acceptance. Not pursued.

## Summary: surviving candidates (napkin gain vs HEAD's measured run)
1. Mixed-step CPU overhead: full-graph or leaner eager path — **~10-20%**.
2. Pipelined host all-reduce (+ extend to prefill sizes) — **~6-7%**.
3. GEMM + all-reduce overlap (N-chunked / fused) — ~4-5% (after 2).
4. Fewer mixed steps by merging prefills — ~5-10% (overlaps with 1).
5. Mixed-step async (no blocking seq_lens sync) — ~2-4% (overlaps with 1).
6. Decode W8A16 efficiency — ~4%.
7. in_proj_ba kernel — ~1.7%.
8. 3 drafts — ~2% now, more after 2/3.
9. Small-kernel fusion, lm_head kernel, lazier GDN commit — ≤ 1.5% each.
