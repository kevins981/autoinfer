# T1 sweep, step 4: knob by knob

Gains are napkin, against BASE's measured 143.2 s per trial (635.9 tok/s), using the calibrated
buckets of 3-calibration.md (2,140 decode steps of 38.5 ms, 312 mixed prefill steps of 147 ms).
"+x%" = throughput gain if the saving is fully realized.

## Knob 1: Keep / Recompute
| site | candidate | napkin | verdict |
|---|---|---|---|
| k,v (16 attn layers) | keep (baseline) | — | already optimal; capacity not binding at c=16 (KV usage 84% of the 5 GiB budget, no preemption seen) |
| S[r,l] GDN state | keep fp32 (baseline) vs lower precision | 1.2 GB/step less | **illegal** (changes the math: state dtype is part of the model, approx forbidden) |
| S[r,l] | DeltaLog / Replay (02 §4.3 row B): keep S[t-L] + L inputs, flush every L steps | write-back 1.2 GB/step -> ~0.15: -1.1 ms/step -> 2.3 s, **+1.6%** | valid (replay is the same sequential recurrence, exact if replayed in the same order); custom kernel + state bookkeeping in vLLM's mamba cache + MTP rollback: high cost. Low rank |
| logits | keep across steps | Rule 2 (contains t) | illegal |
| prefix (chat template) | prefix caching (on, required) | hit rate 0% on this benchmark | nothing to gain |
| draft hidden states | MTP already keeps target hidden h | — | done |

## Knob 2: Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| argmax over v after vocab split | **max-merge before gather** (05 §2 Rule 2 with combine tag max): gather (max, idx) per row instead of full logits; for the drafter always (greedy), for the target when every request in the batch is greedy without logprobs/penalties (generic, falls back otherwise) | AG 1.1 ms/step decode, ~1.7 ms/mixed step -> ~2.5 s, **+1.7%** | exact (same argmax incl. lowest-index tie-break). Rank mid |
| lm_head for drafter | FP8 copy (ruling 1) | draft lm_head bytes /2: 2 x 0.68 ms = -1.4 ms/step -> 3.0 s, **+2.1%** | allowed; vLLM option `draft_lm_head_quantization: "fp8"` exists (MTP inherits it): a flag test |
| lm_head for drafter | **reduced vocab (e.g. 32K most frequent tokens from generic, non-benchmark text) + FP8, replicated on both GPUs (no gather)** | per draft step 1.36 ms + ~0.25 ms AG -> ~0.17 ms: -2.9 ms/step -> 6.2 s, **+4.5%**, minus acceptance loss (each 1% of acceptance length = -1%) | allowed (ruling 2); needs a token list from generic text + code. Rank high |
| AR + residual add + RMSNorm | fused (flashinfer allreduce_fusion) | ~64 norm launches x 3.6 us = 0.23 ms/step | small, and the fused AR kernels need P2P/IPC. Drop |
| GDN in_proj qkvz + ba | merge GEMMs | a few small launches | small. Drop |
| distributivity over r (LoRA-like) | n/a: dense, single tenant | — | n/a |

## Knob 3: Speculate
| site | candidate | napkin | verdict |
|---|---|---|---|
| x[t+1] (token) | MTP chain k=2 (baseline) | acceptance length 2.345 | — |
| x[t+1] | **k=3** | tokens/step 2.345 -> ~2.7 (if p3 ~0.35-0.40); step cost +~5 ms with the current draft head (M 48->64: AR bytes +33% = +2 ms, 3rd MTP step incl. bf16 lm_head ~ +2 ms, GDN/attn +0.5) = +13% -> net ~+3%; **with the cheap reduced-vocab draft head and a faster AR: +~2.5 ms -> net ~+9%** | exact; flag. Rank high *combined* with the draft head |
| x[t+1] | k=1 | tokens/step 1.78; saves 1 MTP step + 1/3 rows | loses (−~15%). Drop |
| x[t+1] | tree (EAGLE-2 style) | N=48 << ridge 82 leaves little slack here (ridge is low on this GPU) | MTP in vLLM is chain-only; GDN state + tree masks: big build. Drop for this budget |
| x[t+1] | retrained / better drafter | acceptance +x% -> +x% | allowed with non-benchmark data, but training cost far beyond our GPU budget. Drop |
| sel (experts) | n/a (dense MLP) | — | n/a |
| y[l] (layer output) | high cardinality, p=0 | — | fails gate 2 |

## Knobs 4 & 5: Split / Merge and Place
| site | candidate | napkin | verdict |
|---|---|---|---|
| m, h, hv, v split over 2 GPUs | TP=2 (baseline) | 128+ ARs/step | reference |
| l | PP=2 (32/32 or asymmetric 33/31 + lm_head) | decode: each GPU must read its 12.5 GB once per micro-batch; 2 micro-batches -> 2 x 15.6 = 31 ms vs TP 17.5 + 7.3 = 25 ms; 1 micro-batch -> 50% idle | **loses at decode** (shared-read rule: weights re-read per micro-batch while memory-bound). Wins on mixed steps (no 65 ms of AR), but decode is 57% of time. Drop |
| r | DP=2 | 27 GB FP8 + 2.5 GB lm_head > 24 GB | capacity: illegal |
| hybrid (TP for attn, PP for MLP, etc.) | — | dominated by 05 §4.1 (alternating cuts cost 2L crossings) | drop |
| **d of GEMM_out (AR is elementwise in d)** | split GEMM_out into chunks along d, AR of chunk c overlaps GEMM of chunk c+1 (Split d + Window) | decode: hide up to min(GEMM_out, AR) ~ 26-46 us per AR -> up to ~3 ms/step -> 6 s (+4%); mixed: ~200 us per AR -> 13 ms/step -> 4 s (+3%) | exact. NCCL kernel competes for SMs; CUDA-graph + 2 streams; medium-hard. Rank mid, after AR itself is fast |
| **r/i of a mixed step (2 micro-batches)** | DBO: AR of micro-batch A overlaps GEMMs of B, only on mixed (compute-bound) steps | mixed step: 92 ms GEMM + 65 ms AR serial -> ~max(92+5, 65) ~ 100 ms: -47 ms x 312 = 15 s, **+11%** (shrinks to ~+3% if AR gets 3x faster first) | exact. Decode steps must not do this (memory-bound: weights re-read, 06 §3). vLLM has DBO (`use_ubatching`, prefill/decode token thresholds) but built for EP all-to-all: needs a check whether TP all-reduce overlaps. Rank mid-high |
| place AR transport | **faster all-reduce: NCCL protocol/channels (Simple/LL128, more channels), or FlashInfer PCIe-IPC all-reduce (in this vLLM build, opt-in)** | decode AR 46 -> ~27 us: -2.6 ms/step -> 5.5 s; mixed AR 481 -> ~130 us: -47 ms/step -> 15 s; together up to **+14%** | exact (2-rank sum is order-free; still bf16 comm). Cheapest test of all (env vars). **Rank 1** |
| place draft lm_head | replicate (with reduced vocab) | see knob 2 | in rank 2 |
| place KV on host | capacity not binding | — | drop |

## Knob 6: Window
| site | candidate | napkin | verdict |
|---|---|---|---|
| rows of all requests | continuous batching (baseline, M~42) | — | done |
| prefill chunk size / mixing | chunked prefill (baseline); bigger or smaller chunks | total prefill FLOPs fixed; mixed steps already carry the decode rows | no napkin gain beyond the GEMM-efficiency and AR items. Drop |
| CPU work vs GPU | async scheduling / fewer host syncs | GPU idle 1.3 ms/decode step, 6.4 ms/mixed step -> ~4.8 s, +3% at most | check whether async scheduling is already on (V2 runner). Low-mid |
| draft steps vs anything | MTP draft step k+1 depends on k; target step t+1 on verify of t | — | no independent partner on the same GPU except AR overlap (above) |

## Implementation-layer gaps the framework doesn't price (calibration findings)
| site | candidate | napkin | verdict |
|---|---|---|---|
| prefill GEMMs at 46% of bf16 peak (Marlin) | **W8A8 block-FP8 kernel (checkpoint's own scheme, allowed) or a better dequant GEMM for M >= ~128** | 92 -> ~55 ms per mixed step: -37 ms x 312 = 11.5 s, **+8.7%** | FP8 MMA with fp32 accumulate on Ada is ~the same 165 TFLOP/s as bf16, so the gain is kernel efficiency, not FLOP rate (fp16-accumulate FP8 at 330 would need a ruling: likely excluded by "fp32 accumulation across K-blocks"... within-block fp16 accumulate is the grey zone). Microbenchmark vLLM's Triton `w8a8_block_fp8_matmul` vs Marlin at M=256..2048 first. Rank high |
| decode GEMMs at ~80% BW | better M=48 kernel (Marlin tuning, other FP8 weight-only kernels) | 17.5 -> ~14 ms: -3.5 ms x 2,140 = 7.5 s, +5% | hard kernel work. Mid-low |
