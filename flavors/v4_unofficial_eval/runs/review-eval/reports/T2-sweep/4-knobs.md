# T2 sweep, step 4: knob by knob

Gains are napkin against BASE's 143.2 s (calibrated model, 3-calibration.md): saved seconds and
% of 143 s. Calibrated efficiency ratios from step 3 are applied to "ideal" kernels (e.g. a new
memory-bound kernel is assumed to reach ~80% of DRAM BW, a tuned link ~60% of 25 GB/s).
Buckets: decode steps ~2,130 (38.5-39.3 ms each), mixed steps ~370, all steps ~2,500.

## Knob 1: Keep / Recompute (data nodes across steps)
| site | candidate | napkin | verdict |
|---|---|---|---|
| GDN state S[r,l] (fp32, overwritten each step: 2.4 GB/step traffic) | keep in bf16 | -1.2 GB/step | **pruned**: changes precision of the GDN math (contract: attention/GDN math unchanged); approx row, not allowed here |
| same | ReplaySSM row B: keep S[t-L] + last L inputs, write S only every L steps | saves ~half the writes: ~1.2 GB/step -> ~1.5 ms x 2,130 = 3.2 s (2.2%) | valid (fp reordering); kernel work in the MTP GDN kernel + rollback logic; medium-high effort, mid gain: **kept, rank low** |
| KV cache k,v (fp8, append-only) | already kept; nothing cheaper exists (Rule 1/2) | 0 | baseline row |
| logits (target, per step) | keep nothing: only the argmax is read at temperature 0 (see Rewrite R1) | — | handled by R1 |
| KV/mamba block budget (5 GiB, block 1600 tokens, "align" mamba cache) | more KV budget so no request waits | server log shows "Waiting: 1-2" with 14-15 running at times; unclear whether blocked by KV or just stats timing | **untested question**; cheap to check from logs of any run |

## Knob 2: Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| R1 logits gather -> argmax (target verify + 2 draft steps) | split-v reduction: each rank computes (max, argmax) over its vocab half, gather 2 numbers per row, combine with lowest-index tie-break; fast path when every request in the batch is greedy (a request property, not benchmark detection) | removes 1.67 ms of AllGather + ~0.1 ms of elementwise per step: ~1.6 ms x 2,500 = **4.0 s (2.8%)** | exact; **kept** (medium effort: sampler + rejection sampler + drafter paths) |
| R2 draft lm_head kernel (cutlass wmma, grid 8 CTAs, 1.37 ms at M=16) | (a) better bf16 kernel at full BW (~0.75 ms); (b) FP8 draft lm_head (ruling 1: allowed in drafter only) ~0.4 ms | (a) 2 x 0.6 = 1.2 ms/step -> 3.0 s (2.1%); (b) 2 x 0.97 = 1.9 ms -> **4.9 s (3.4%)** | exact for (a), drafter-only change for (b) (acceptance may move slightly); **kept** |
| R2' target lm_head kernel (ampere bf16 128x64, 1.59 ms at M=48) | kernel choice tuned for M<=64 (~0.8 ms) | 0.8 ms x 2,500 = 2.0 s (1.4%) | exact (target precision unchanged), **kept** (cheap if a cuBLAS/heuristic choice) |
| R3 reduced-vocab draft lm_head (ruling 2) | top-32K tokens from non-benchmark text | lm_head draft 0.64 -> 0.08 GB: ~2.5 ms/step -> 6 s (4%) minus acceptance loss | needs generic token stats + acceptance check; **kept as alternative to R2b**, higher risk |
| R4 W8A8 (checkpoint's own dynamic per-token-group FP8) for **mixed/prefill** GEMMs | FP8 tensor cores instead of bf16 Marlin (which also re-reads weights per 64-row slice) | if FP8 fp32-acc peak = 330 TF: Marlin 0.19 -> ~0.09 ms/token: 104.5K x 0.10 = **10 s (7%)**; if 165 TF: ~0-3 s | allowed (task_definition), needs GSM8K; **kept, value hinges on the worker's GEMM microbenchmark** |
| R4' W8A8 for decode GEMMs (M = 48, memory-bound) | only wins if the FP8 kernel streams weights faster than Marlin (Marlin 650 GB/s in-server) | 18.6 -> ~15 ms if 820 GB/s: 3.6 ms x 2,130 = 7.7 s (5%) | **kept, uncertain** (kernel quality at small M) |
| R5 fuse all-reduce + residual RMSNorm | removes 132 norm launches (0.48 ms/step) | <0.5 ms/step -> ~1 s | **pruned**: small; NCCL can't fuse; only with a custom all-reduce (P5) |
| R6 dedup across the 3 verify rows' state reads | already done by the MTP GDN kernel | 0 | baseline |

## Knob 3: Speculate
| site | candidate | napkin | verdict |
|---|---|---|---|
| S1 token edge, chain length k: 2 -> 3 | acceptance per position 0.78, 0.56 -> extrapolated ~0.40 at pos 3: tokens/step 2.345 -> ~2.75 (+17%) | cost now: +1 draft step (~2.6 ms) + M 48->64 (Marlin memory-bound +~1 ms, AR +0.8 ms, GDN/attn +0.5) = +4.9 ms (+13%): **net ~+3%**; after R1+R2 (draft step ~1 ms): +3.3 ms (+9%) -> **net ~+7%** | flag-only; **kept**; best tested after R1/R2 |
| S1' k = 1 | 2.345 -> 1.78 tokens/step, step -3.5 ms | net loss (-15%) | **pruned** |
| S2 tree speculation (top-2 at depth 1) | not supported by vLLM's MTP path; M grows | rows are not free here (ridge ~82, already at M=48 with mixed costs) | **pruned**: high effort, small free slack |
| S3 speculate the target's verdict to start draft s=1 early (overlap drafter with target lm_head/verify) | the drafter's input depends on the accepted token (cardinality small, p(all accepted) ~0.44) | could hide ~1-2 ms/step but wastes work 56% of the time; same GPUs, same DRAM (Gate 1 fails: no orthogonal resource) | **pruned** (Gate 1) |
| S4 layer-output / hidden edges | high cardinality, p = 0 (exact) | — | **pruned** (Gate 2) |

## Knobs 4/5: Split / Merge and Place
| site | candidate | napkin | verdict |
|---|---|---|---|
| P1 parallelism: TP=2 (h, m, v split) vs PP=2 (l split) | PP removes 135 all-reduces per forward but needs 2 micro-batches to fill the pipe; decode is memory-bound so each GPU reads its weights once per micro-batch: round = 2 x ~17 ms + lm_head ~ 36-37 ms vs TP 38.5 ms; prefill gains the AR time (0.1 ms/token) | decode ~0-4%, mixed ~+7%; but vLLM V1 PP + MTP spec decode + hybrid GDN support is doubtful and asymmetric lm_head (1.27 GB on one stage) needs a 33/31 split | **pruned for now**: high effort, uncertain support; revisit if AR can't be fixed |
| P1' DP=2 | 28.75 GiB weights > 24 GB | — | **infeasible** |
| P1'' hybrid: TP for body, vocab split lm_head | that is BASE | — | baseline |
| P4 all-reduce transport (Place of the reduction buffers / NCCL protocol, channels) | 2 channels, LL protocol, SHM: 60 us per 0.49 MB (8 GB/s); target ~30 us; prefill 11-14 GB/s -> ~20 GB/s | decode 135 x 30 us = 4 ms x 2,130 = 8.5 s; mixed 0.10 -> 0.06 ms/token = 4.2 s: **~12.7 s (8.9%)** upper bound | env vars / NCCL config (generic default, ruling 3): **kept, cheapest test**; microbenchmark running (leader-2.w1) |
| P5 custom all-reduce through host-mapped pinned memory (one-shot: write own half, read peer's) | latency ~1 PCIe round trip + 0.49 MB / 25 GB/s ~ 25 us | similar to tuned P4; adds fusion (R5) | **kept as follow-on** if P4 tops out high |
| P6 split payload along rows (r or i) and overlap the AR of piece 1 with GEMMs of piece 2 (Window mech. 2) | mixed steps are compute-bound: weights re-read per piece costs little; hides up to the 0.10 ms/token AR | up to 10.5 s (7.3%) in mixed steps; decode: re-reading weights per piece costs +~9 ms > AR saved: **loses** | **kept for mixed steps only**, high effort (two streams per layer) |
| P7 asymmetric TP split (alpha != 0.5) | ranks are symmetric (lm_head split by vocab, same state) | 0 | **pruned** (symmetry proof, 05 §4.1) |
| P8 place GDN state / KV on host | capacity is not binding at c=16 | 0 | **pruned** |

## Knob 6: Window (batching and overlap)
| site | candidate | napkin | verdict |
|---|---|---|---|
| W1 mixed steps run eagerly (largest captured graph 96 tokens); E ~35 ms CPU/launch overhead per mixed step | capture graphs (piecewise/FULL "prefill" graphs) up to ~512-2048 tokens, or torch.compile + piecewise graphs | up to 370 x 35 ms = **13 s (9%)** | flags (`--compilation-config` capture sizes / `max_cudagraph_capture_size`); memory must pass 128K test; **kept, cheap test** |
| W2 GPU idle inside decode step (1.3 ms: CPU prep between target and draft graphs) | async scheduling / capture drafter + prep in the same graph | 1.3 ms x 2,500 = 3.3 s (2.3%) | check `--async-scheduling` support with MTP; **kept, cheap test** |
| W3 merge draft steps' lm_head reads with the next target step | serial edge (s=2 needs s=1's token) | — | **pruned** (dependency rule) |
| W4 chunked prefill split of long prompts across steps | already merged with decode rows; splitting more just moves time | ~0 | **pruned** |
| W5 torch.compile fusion of small ops in the decode graph (norms, act, conv: ~1.3 ms) | mode NONE today | <=1 ms/step -> ~2 s (1.5%); startup cost | **kept low** (combine with W1 test) |

## Summary of kept candidates (napkin, % of 143 s)
W1 mixed-step graphs 9% | P4 NCCL tuning <=9% | R4 W8A8 prefill 0-7% | S1 k=3 3-7% | R2b FP8 draft lm_head 3.4% (R2a 2.1%) | R1 split-argmax 2.8% | R4' W8A8 decode 0-5% | P6 AR/compute overlap (mixed) <=7% | W2 async 2.3% | ReplaySSM 2.2% | R2' target lm_head 1.4% | W5 compile 1.5%.
