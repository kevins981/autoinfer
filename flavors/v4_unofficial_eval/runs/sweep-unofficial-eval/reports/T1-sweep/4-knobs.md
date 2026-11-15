# T1 sweep, step 4: knob by knob over the graph's sites

Baseline for all gains: BASE official run 143.2 s (calibrated split in 3-calibration.md: ~2,150 decode steps x 38.5-40 ms,
~312 prefill-containing steps x ~195 ms). Gain % = seconds saved / 143.2 s. Gains of different rows are NOT additive
where they attack the same time (noted). "drop" = pruned, with the reason.

## Knob 1: Keep / Recompute (data nodes kept across steps)
| id | site | candidate | napkin | verdict |
|---|---|---|---|---|
| K1 | k,v[r,l,j] | keep k,v (current) vs keep x / recompute | KV read per decode step ~7 MB: recompute costs >> | drop: current row wins by orders of magnitude |
| K2 | S[r,l,h] fp32 GDN state, in-place overwrite (current, Row A) | Row B (ReplaySSM: keep S[t-L] + last L steps' k,v,g,b; flush every L) | state traffic 50 MB/layer/step -> ~25(1+1/L) MB: saves <= 1.2 GB/step = 1.2 ms x 2,150 = 2.6 s (**1.8%**); new kernel + state plumbing with prefix-cache align mode | keep (low rank: big effort, small gain) |
| K2b | S dtype | bf16 state | halves state bytes: ~1.8 ms/step | drop for exactness: `mamba_ssm_dtype float32` is the model's numerics -> `approx` row; would need a ruling + accuracy gate; not pursued |
| K3 | mamba prefix-cache block copies (precopy/postprocess ~32 us/step) | disable prefix caching | 0.07 s | drop: <0.1% |
| K4 | KV/state capacity | bigger KV budget | 0 preemptions, usage <= 84% in the official log | drop: capacity not binding at c=16 |

## Knob 2: Rewrite
| id | site | candidate | napkin | verdict |
|---|---|---|---|---|
| R1 | argmax over v after vocab-parallel lm_head (combine tag = (max, lowest index)) | local argmax per rank + all-gather of 2 numbers/row instead of all-gathering z | removes target AG 0.96 ms + draft AGs 2x0.33 ms per decode step (+~1 ms per prefill step): 3.8 s (**2.6%**). Bitwise-exact for greedy rows; must fall back when a request needs logprobs/temperature>0. Draft half exists as a flag (`use_local_argmax_reduction`) | **keep** (draft half = flag test; target half = code in the rejection sampler) |
| R2 | prefill-row GEMMs (compute-bound, 437 rows >> ridge 82) | checkpoint's W8A8 (per-token-group-128 FP8 activations) with FP8 mma, fp16 accumulator inside each 128-K block, fp32 across blocks | 330 vs 165 TF: Marlin in prefill steps 91.9 -> ~46 ms: 312 x 46 = 14 s (**~10%**). With fp32 accumulation the FP8 path is also 165 TF -> only kernel-efficiency gains (~0-2%) | keep, **risky**: e4m3 x e4m3 products reach 2e5 > fp16 max, so in-block fp16 accumulation can overflow; needs a kernel (vLLM's triton block-FP8 uses fp32 acc) + GSM8K-250 A/B |
| R3 | decode-row GEMMs (Marlin, memory-bound) | same math, better kernel/config for M=48 (Marlin reaches 0.80 TB/s on the big GEMMs, 0.67 overall) | 18.6 -> ~14.8 ms at 85% BW: 8.2 s (**5.7%**) | keep (implementation: per-shape tuning, fusing small GEMMs, fewer tails) |
| R4 | all-reduce + residual add + RMSnorm | fuse into the AR kernel (`fuse_allreduce_rms` off now) | 130 x ~3.6 us = 0.47 ms/step: 1 s (**0.7%**) | keep as add-on to a custom AR (X1) |
| R5 | ~400 small elementwise kernels; eager prefill steps | torch.compile (compile mode is NONE in BASE) / CUDA graphs for prefill | decode ~0.5 ms/step + prefill launch idle (15 ms in nsys, real part unknown): ~1-3 s (**1-2%**) | keep (cheap flag test; why is compile off? startup?) |
| R6 | draft lm_head (2 x 1.37 ms/step, same W_lm as the target) | (a) FP8 draft head (`draft_lm_head_quantization=fp8`, ruled allowed); (b) reduced draft vocab V' from a non-benchmark frequency source (ruled allowed) | (a) saves ~1.37 ms/step: 2.9 s (**2.0%**), acceptance ~unchanged; (b) V'=32K saves ~2.4 ms/step: 5.1 s (**3.6%**) minus the acceptance loss on tokens outside V' | keep (a: flag test; b: needs a generic frequency list, e.g. BPE merge order as a proxy) |
| R7 | same tensor W_lm read in 3 windows per step | dedup the read (one window) | impossible: target sample -> draft 1 -> draft 2 is a serial chain | drop |

## Knob 3: Speculate (two gates per edge)
| id | edge | candidate | napkin | verdict |
|---|---|---|---|---|
| S1 | x[t+1] (token, low cardinality) | 3 MTP drafts instead of 2 (M 48 -> 64) | tokens/step 2.345 -> ~2.75 if p3 ~0.40 (+17%); step cost +~7 ms at BASE costs (Marlin +0.9, AR +2.7, GDN +0.6, 3rd draft step ~2.3, target AG +0.3) = +17% -> **~0%** now; **+5-6%** once the draft step costs ~0.6 ms (R1, R6) and AR is fast (X1) | keep, conditional on R1/R6/X1; riskiest: p3. Cheapest test: k=3 server, read per-position acceptance (~3 GPU-min) |
| S2 | x[t+1] | tree drafts (top-2 at depth 1) | +~8% tokens/step for +33% rows; GDN tree verification needs branching state (rollback-free verify) | drop: implementation far beyond budget; chain first |
| S3 | x[t+1] | better/retrained drafter | acceptance gains unknown; needs training compute and non-benchmark data | drop for this budget (80 GPU-min) |
| S4 | all-reduce result, layer outputs | speculate a continuous vector | p ~ 0 (gate 2) | drop |
| S5 | `num_speculative_tokens_per_batch_size` (exists) | adapt k to batch size | at c=16 equals S1 | merged into S1 |

## Knobs 4/5: Split / Merge and Place
| id | site | candidate | napkin | verdict |
|---|---|---|---|---|
| X1 | all-reduce over p (sum, 491 KB in decode, ~4.5 MB in prefill steps) | faster AR: NCCL protocol/channels tuning, or a custom one-shot AR through pinned host memory mapped into both GPUs (no P2P needed; each GPU writes its partial once and reads the peer's once) | decode 60 -> ~25 us: 135 x 35 us = 4.7 ms/step -> 10 s (**7%**); prefill ~520 -> ~200 us (at ~20 GB/s): 41 ms/step -> 12.8 s (**9%**). Total **~15%** | **keep, top**. Riskiest: achievable host-path bandwidth/latency; cheapest test: microbenchmark (worker t1-w1 running) |
| X2 | layers | PP=2 instead of TP=2 | decode: each GPU reads its 12.2 GB once per micro-batch; 16 seqs need 2 micro-batches (M=24) to keep both busy -> ~31 ms Marlin + GDN + lm_head on one GPU ~42 ms vs TP 40 (and ~35 once X1 lands); prefill steps win (no ARs) | drop: dominated by TP + X1 at c=16 decode (weight bytes per token double; framework's DualPipe-at-small-batch case) |
| X3 | split ratio | asymmetric TP/PP split | both ranks do equal work in TP | drop (symmetric optimal when T_other equal) |
| X4 | AR -> reduce-scatter + all-gather (sequence parallel) | same link bytes; norms on half rows | saves ~0.2 ms/step norms | drop (no link-byte change) |
| X5 | replicate the vocab-parallel embedding (removes 3 ARs per step) | +1.27 GB per GPU | 0.18 ms/step: 0.4 s (0.3%) | drop (tiny; memory) |
| X6 | drafter TP=1 / replicated | lm_head 2.5 GB on one GPU per draft step | slower | drop |
| X7 | KV/state offload to host | capacity not binding | 0 | drop |

## Knob 6: Window (merge for shared reads; overlap orthogonal resources)
| id | site | candidate | napkin | verdict |
|---|---|---|---|---|
| W1 | prefill-containing steps: AR (link) vs GEMM (compute) | split the step's rows into 2 half-batches, stagger by one window (dual-batch overlap): GEMM(half 2) runs while AR(half 1) is in flight. Weights read twice, but prefill GEMMs are compute-bound (437 rows >> 82) so the extra read is hidden | hides up to ~min(compute, comm) per layer: 66 ms -> ~10 ms: up to 18 s (**~13%**) with BASE's AR; ~4% after X1 | keep (big but complex: two streams in the eager prefill path; NCCL must not steal much SM/HBM) |
| W2 | decode steps, same overlap | half-batches of 8 seqs | doubles weight reads (+~15 ms) to hide <= 9.8 ms | drop (framework: loses below the ridge) |
| W3 | inside one GEMM->AR pair | split d (output columns) of o_proj/out_proj/down into chunks; AR chunk c while computing chunk c+1 (no extra weight reads) | decode: hides <= ~25 us per AR -> <= 3 ms/step; prefill similar to W1 but finer | keep as alternative to W1 (contention with Marlin's 128 SMs is the risk) |
| W4 | CPU step preparation vs GPU execution | async scheduling (overlap CPU prep of step t+1 with GPU step t) | GPU idle 1.4 ms/decode step in nsys (inflated): <= 3 s (**<=2%**) | keep (cheap flag test if supported with MTP) |
| W5 | GDN state stream || GEMM weight stream | both HBM | drop (same resource) |
| W6 | prefill chunk size | fixed total prefill FLOPs and AR bytes | drop (no byte/FLOP change; latency only) |

## What the framework could not express (or mispriced) here
- The link is not a single `Link_BW`: the measured AR runs at ~8 GB/s (decode) and ~4.3 GB/s (prefill) against a 25 GB/s spec. Protocol overhead (NCCL LL = 50% payload) and host-staging are implementation constants, but they are 3x, i.e. structural for ranking. I price X1 with a measured efficiency once t1-w1 reports.
- Launch/latency-bound small kernels (~1.5 ms/decode step) have no FLOP/byte cost; the framework has no term for them (needs a per-kernel latency floor ~2-4 us).
- The ridge point for Marlin is ~82 rows (bf16 compute on fp8 weights), not 330: the 01 file's 4090 example assumes FP8 tensor math.
- The accumulation-precision rule (fp16 vs fp32 accumulate halving F on GeForce) is a hardware detail that decides R2.
