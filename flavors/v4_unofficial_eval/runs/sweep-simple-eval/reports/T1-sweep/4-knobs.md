# T1 sweep, step 4: knob by knob

Gains are napkin estimates against BASE's measured time (3-calibration.md): decode step 38.5 ms
(68% of benchmark time), mixed step 146.8 ms (32%). "Overall" = share-weighted saving of total
time (throughput gain ~ 1/(1-saving) - 1).

## Knob 1: Keep / Recompute
| site | candidate | napkin | verdict |
|---|---|---|---|
| k,v (16 attn layers) | keep (BASE) | 0.42 ms/step | keep; nothing to gain (attention is 1% of a step) |
| GDN S[r,l] fp32 | in-place keep (BASE, Row A) | 2.95 ms vs 2.4 floor | Row A is at 80% of floor |
| GDN S with MTP verify | Row B (checkpoint + replay inputs, write S once per L steps) | saves up to the write half, ~1.2 ms/decode step = 2.1% overall | **candidate K-B**, but custom kernel + rollback logic: high effort for ~2% |
| GDN S in bf16 | lower-precision keep | -1.2 ms | **illegal** (changes the math; state dtype is config fp32) |
| prompt prefix | prefix caching (on) | ShareGPT prompts share only the chat template | nothing to gain |
| logits | keep vs recompute | no future readers | dead (Rule 1) |

## Knob 2: Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| argmax over split v (target + draft) | **distributed argmax**: per-rank (max, idx), combine with argmax-merge instead of all-gathering logits (exact; ties -> lowest index) | removes 0.95 + 0.39 ms per decode step, ~1.7 ms per mixed step: **~2.4% overall** | **candidate R-ARGMAX**; fallback to full gather when a request needs logprobs / non-greedy sampling |
| residual add + RMSNorm after AR | fuse into AR kernel (AR+norm) | saves 132 x ~3.5 us = 0.46 ms/decode step (1.2% decode) | small; comes free with a custom AR (I-AR) |
| SiLU*mul + Marlin input | fuse | ~0.2 ms | too small |
| W8A8 (checkpoint's own scheme) for prefill GEMMs | rewrite bf16xFP8 into FP8xFP8 per-128 block | with fp32 accumulate same 165 TF peak as bf16 -> gain only from kernel efficiency (Marlin 65%); with fp16 accumulate inside a 128-K block (allowed) 330 TF -> 59 -> ~30 ms per mixed step = **~6% overall** | **candidate R-W8A8**; riskiest: fp16 overflow of a 128-term e4m3 partial sum (448^2 x 128 >> 65504) -> needs a range argument or a guard |
| draft lm_head FP8 (ruling 1) | 1.27 GB bf16 -> 0.64 GB FP8 | -0.65 ms per draft lm_head; ~1.1% overall | **candidate R-DLM8** (small, low risk) |
| reduced-vocab draft lm_head (non-benchmark token stats) | 248K -> 32K rows | -1.2 ms/decode step but acceptance loss unknown | candidate, needs generic token-frequency data; lower priority |

## Knob 3: Speculate
| edge | candidate | napkin | verdict |
|---|---|---|---|
| token edge | MTP k=2 (BASE): 2.345 tok/step | — | baseline |
| token edge | **MTP k=3** (more drafts) | acceptance pos2 ~0.36 (guess: conditional 0.65) -> 2.71 tok/step (+15.5%); cost: rows 48->64 (Marlin memory-bound ~+0.5 ms, AR bytes +33% ~+2.5 ms at today's NCCL, GDN MTP kernel +~0.7 ms), +1 draft step (~1.4 lm_head + 0.4 AG + 0.5 layer + AR) ~ +2.5 ms -> ~44.5 ms: decode tok/s x 1.155 x 38.5/44.5 = **~0%..+5%**; much better if comm gets cheaper | **candidate S-K3**: cheapest test of all (serve flag) |
| token edge | MTP k=1 | 1.78 tok/step at ~33 ms -> loses | dropped |
| token edge | tree drafts (top-2 at pos 1) | needs tree attention + GDN tree states: GDN state is in-place overwrite (02 §4.4 trap) | dropped: GDN tree verification needs per-branch state, high effort |
| layer edges / GDN state edge | — | high cardinality, W carries l | dropped (Gate 1 + Gate 2) |
| draft step edge x̂_1 -> x̂_2 | parallel drafting | MTP head is sequential by training | dropped |

## Knob 4/5: Split and Place
| split | candidate | napkin | verdict |
|---|---|---|---|
| heads/m (TP=2, BASE) | 130 ARs per decode step | — | baseline |
| l (PP=2, 32+32 or asymmetric 33/31 for lm_head) | no ARs; decode needs 2 microbatches => each GPU reads its 12.2 GB twice per round: ~2 x (18.6 + 3) ~ 44 ms per round (vs 38.5) = -13% decode; mixed step ~59+10 = 69 ms (vs 147) = +53% on 32% of time | net **~+7% at today's comm**, **negative** once AR is fixed (I-AR); MTP + PP support in this vLLM uncertain | conditional candidate P-PP; dominated by I-AR if I-AR reaches its napkin |
| r (DP=2, each GPU full model) | 24.3 GB FP8 + lm_head 2.5 GB > 24 GB | **infeasible** (capacity) | dropped |
| attention h split, MLP m split | TP variants with sequence-parallel (RS + AG) | same link bytes; norm on half the rows (-0.2 ms) | dropped (no link saving) |
| v (lm_head, vocab-parallel) | keep split; change the combine (see R-ARGMAX) | | via R-ARGMAX |
| Place: weights partly on CPU / peer | PCIe 25 GB/s << 1 TB/s | dropped: everything waits on weights |
| Place: KV offload | KV is small (5 GiB budget is enough) | dropped |

## Knob 6: Window
| windowed nodes | candidate | napkin | verdict |
|---|---|---|---|
| AR(l, half A) || GEMM(l+1, half B), mixed steps (two half-batches staggered, DualPipe-style inside TP) | mixed step compute-bound, so reading weights twice is ~free: step ~ max(59+6, 62) + ~10 ~ 80 ms (vs 147) | **~14% overall** at today's comm, ~4% after I-AR | **candidate W-DBO** (high effort: two streams per layer, NCCL on a side stream, eager path only) |
| same for decode steps | memory-bound: second weight read +18.6 ms > 7.75 ms AR saved | dropped (Shared-Read Rule: loses at N << ridge) |
| batching 2 prompt chunks per step | per prompt token cost constant (compute-bound + AR ∝ rows); saves one decode iteration riding per pair | ~1% | dropped (and TTFT cost) |
| draft lm_head || target AR | different resources but serial dependency | — | dropped |

## Below the framework: implementation efficiency (found by calibration mismatches)
| site | candidate | napkin | verdict |
|---|---|---|---|
| all-reduce transport (130/step) | **faster AR**: NCCL protocol/channel tuning (Simple/LL128, SHM via cudaMemcpy, more channels) or a custom host-memory (pinned, zero-copy or copy-engine) AR, or flashinfer's PCIe IPC AR (opt-in in this vLLM) | at ~napkin: decode 7.75 -> ~4 ms (-3.7), mixed 62.5 -> ~16 ms (-46): **~11.5% of total time => ~+13% tok/s**; at 2x improvement ~+8% | **candidate I-AR: #1**. Cheapest test: AR microbenchmark across variants (running: worker leader-1.w1) |
| Marlin decode BW 67% | better FP8 weight-only kernel / tuning (e.g. split-k, larger tiles for 16 MB out_proj) | 18.6 -> 14 ms at 90% BW: -4.6 ms decode = ~8% overall | candidate I-MARLIN; high effort, uncertain |
| eager small kernels in mixed steps; no torch.compile | enable compilation (inductor fusions) / CUDA graphs for larger token counts | ~3-5 ms per mixed step = ~1%; decode ~0.5 ms | candidate I-COMPILE, low priority (startup time, risk) |
| GPU idle between steps (1.3 ms decode) | async scheduling / CPU overhead | ~1-2% | check whether async scheduling is on; low priority |
