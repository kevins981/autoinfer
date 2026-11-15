# T3 sweep, step 4: the knobs, site by site, priced against HEAD's measured time

Time model (from 3-calibration.md), per 200-prompt run (56.9 s): decode-only steps ~1230 x 28.8 ms = 35.7 s (63%);
mixed steps ~21 s (37%) = ~106 ms extra per admitted prompt. Prompt mix in the profile: ~50% small prefills
(M <= ~130 rows incl. decode rows), ~30% mid (M 200-500), ~20% large (M 500-1900).
Gain = seconds saved / 56.9 s. "Decode ms" = per decode step (x1230); "mixed ms" = per mixed step (x200).

## Knob 1: Keep / Recompute (data nodes kept across steps)
| site | options | napkin | verdict |
|---|---|---|---|
| GDN state S (fp32, 1.57 MB/req/layer/GPU) | keep S in place (BASE) / S + pending rows, 1 write per step (HEAD replay) / lazy commit every K steps / token history only (Row C) | HEAD at 1.91 ms vs ~1.8-2.4 floor of its keep set. K=3+: write 1/3 instead of 1/2 of S: -0.3 ms decode = 0.6%. Row C: wins only for t < d_k = 128; prompts ~260+ -> loses | lazy K>2: **small (0.6%)**, low priority. Row C pruned (t > 128). bf16 S: forbidden (changes GDN math) |
| KV (fp8), conv state | fixed by contract / tiny | — | none |
| per-layer metadata of the mixed GDN path (masks, index tensors, cu_seqlens views) | recomputed per layer (HEAD: ~24 torch launches per GDN layer) vs computed once per step and kept | see Rewrite R6 (same site) | candidate (CPU) |
| draft-side state (MTP KV) | kept | — | none |

## Knob 2: Rewrite (exact identities)
| id | site | rewrite | napkin | verdict |
|---|---|---|---|---|
| R6 | GDN mixed-batch path, per-layer index/gather/cat/copy ops | **Subgraph dedup across l**: their inputs (spec masks, query_start_loc, state indices) lack the layer index, so compute them once per step; fuse the rest into the layer's kernels | ~1,150 of ~1,700 eager launches per mixed step; CPU 116 ms/step under nsys vs GPU busy 65 ms. If the step is CPU-bound without nsys too: -20..-50 mixed ms -> **7-17%**. If GPU-bound: ~0 | **candidate, uncertain**: needs a no-nsys CPU-vs-GPU measurement of mixed steps |
| R7 | FP8 GEMMs at M > 64 (mixed steps) | Marlin runs i in 64-row passes, each re-reading the weights (W lacks i but the kernel splits i across passes). **Merge the passes**: one kernel with M tiles >= 128 (W8A16) or the checkpoint's W8A8 block-FP8 (allowed) | small mixed: 27.7 -> ~15 ms (weights once) = -12 mixed ms; mid/large: 57-150 ms Marlin -> compute floor 49 ms at 350 rows bf16 (or ~25-30 FP8) = -10..-60 mixed ms. Total **~4-7%** | candidate |
| R1 | RMSNorm before every GEMM | norm scale 1/rms[r,i] factors out of the GEMM (distributivity): GEMM can start on K-chunks before the row's norm is known | enables AR/GEMM overlap along K in decode (see W-sites); alone 0 | enabler only |
| R2 | target lm_head (1.27 GB bf16) | precision fixed (target lm_head stays baseline); vocab subset forbidden for target | — | pruned (contract) |
| R3 | draft lm_head | fp8 + 98K-token subset already (HEAD) | — | done |
| R4 | fused add + RMSNorm after each AR | fold residual add + RMSNorm into the AR kernel's epilogue (AR output is read only by them) | -132 launches x ~3 us = -0.4 decode ms = 0.9% | small, cheap, stacks with B |

## Knob 3: Speculate
| id | site | option | napkin | verdict |
|---|---|---|---|---|
| S1 | next token x[t+1] | 3 drafts instead of 2 | per-position cumulative acceptance 0.78, 0.56 -> 3rd ~0.39 (if conditional ~0.7 holds): tokens/step 2.33 -> ~2.73 (+17%). Step cost: M 48 -> 64 (one Marlin pass still): GEMMs +~0.6 ms, AR bytes +33% (+1.9 ms at HEAD's AR, +0.9 after B), 3rd draft step +0.9 ms, GDN 4 rows (replay kernel max is 3 rows: code change) +0.3 -> 28.8 -> ~32.5 ms (+13%) -> **net ~+3% decode, ~+2% total**; more with a faster AR | candidate, mid; riskiest: the 3rd acceptance; p99 TPOT |
| S2 | tree drafts | needs per-branch GDN states (in-place S) | rollback-free verify needs per-branch replay rows: large change for +5-10% tokens | pruned for cost (not for napkin) |
| S3 | better drafter (retrain MTP) | allowed (non-benchmark data) | acceptance is the multiplier, but training is out of our GPU budget | pruned (budget) |
| S4 | high-cardinality edges (x after AR, layer outputs) | exact verify needs the true value: no gain (Gate 2) | 0 | pruned (framework Gate 2) |

## Knobs 4/5: Split / Merge and Place
| id | site | option | napkin | verdict |
|---|---|---|---|---|
| SM1 | layers across GPUs | PP=2 instead of TP=2 | removes 128 ARs/step (-5.7 ms) but with 2 microbatches each GPU reads its 12.2 GB twice per step (+15 ms); with 1 microbatch 50% bubble | **pruned**: PP wins only when AR time per step > weight-read time per step (5.7 < 15) |
| SM2 | row-parallel GEMM output d (decode) | split d into 2-4 column chunks, AR chunk k on a side stream while GEMM chunk k+1 runs (different weight columns: no re-read) | saves ~min(GEMM chunk, AR chunk) per site: MLP down (58 us GEMM, 42 us AR) ~-21 us, out_proj ~-10 us -> -2.0 decode ms = **~4%** | candidate (stacks partly with B: a faster AR leaves less to hide) |
| SM3 | split r into 2 half-batches (dual-batch overlap) in decode | AR of half A || GEMM of half B | weights read twice at M=24 (memory-bound) -> +15 ms | pruned (framework: DBO loses below the ridge) |
| SM3' | same, in mixed steps (compute-bound GEMMs) | row chunks: AR(chunk k) || GEMM(chunk k+1) -- HEAD has this eager-only (rows >= 192) and graphed steps <= 512 tokens skip it | mid/large mixed: hide ~70% of 44-90 ms NCCL = -30..-60 mixed ms on ~half the prompts -> **~5%** (less after B) | candidate (make it graph-capturable, or route through faster AR) |
| SM4 | prefill length i | smaller prefill chunks | each mixed step has a fixed CPU/launch cost; more steps -> more cost | pruned for throughput |
| P1 | MTP layer placement | TP=1 on one GPU | -4 small ARs (+0.17) vs 2x MTP weights on one GPU (+0.25) and skew | pruned |
| P2 | AR transfer path | SM-driven zero-copy to pinned host (HEAD) vs copy engines (cudaMemcpyAsync) vs NCCL | copy engines don't use SMs (overlap-friendly) but need cross-process sync; SM kernel with enough blocks should reach link speed | covered by B |

## Communication (no knob changes the bytes: bf16 AR is fixed; the cost is how fast the bytes move)
| id | site | option | napkin | verdict |
|---|---|---|---|---|
| **B** | every TP all-reduce | host AR with enough blocks in flight / write-read pipelined; extend it to large tensors (replace NCCL ring at 11 GB/s) | decode: 42.5 -> ~22 us x 135 = **-2.7 decode ms**; small mixed 147 -> ~45 us x 127 = -13 mixed ms; mid/large 329 -> ~150 us x 134 = -24..-50 mixed ms. Total ~3.3 + 1.3 + 3 s = **~13%** | **top candidate**; riskiest: both GPUs' PCIe traffic through host memory reaching ~20 GB/s per direction at once; test: microbenchmark (t3-w1, task 1, running) |

## Knob 6: Window (overlap independent nodes on orthogonal resources)
| id | site | option | napkin | verdict |
|---|---|---|---|---|
| W1 | AR (link) || next GEMM's weight load (HBM -> L2) in decode | weights lack x: loading W[l+1] doesn't depend on the AR. Prefetch the next GEMM's first ~20-40 MB into L2 (72 MB) on a side stream while the AR runs; Marlin then hits L2 | AR window 128 x 42 us = 5.4 ms of idle HBM per step; at most min(AR, next GEMM) hidden: **-3..-5 decode ms (6-9%)** (after B: -2..-3) | candidate, **risky**: L2 residency under Marlin's streaming loads, interference with the AR kernel, graph capture of 2 streams |
| W2 | AR || next GDN layer's state S load (HBM -> L2) | S[l] is kept, independent of x | S per GDN layer 25 MB/GPU: could hide ~25 us per GDN AR; GDN total 1.9 ms -> -0.5..-1 decode ms | subsumed by W1 (same mechanism, smaller target) |
| W3 | mixed step: CPU launch || GPU | the eager GDN/attention parts between 68 piecewise graphs serialize CPU launch and GPU work; full-graph mixed steps or fewer launches | as R6 | same site as R6 |
| W4 | draft steps vs verify | draft u=1 -> u=2 -> next verify: strictly serial | 0 | pruned (dependency) |

## Other (implementation-layer gaps the framework only prices)
| id | site | napkin | verdict |
|---|---|---|---|
| G1 | Marlin decode BW efficiency (76% of spec at M=48) | 88% would save -2.1 decode ms (~4%) | candidate, kernel tuning (HEAD's team already tuned Marlin configs and wrote a Triton W8A16 at 74%; uncertain) |
| G2 | W8A16 Triton mid-M configs (65-512 rows at 1.2-1.45x Marlin, from T1-016's record) | -1.6 small mixed ms | small; part of R7 |
| G3 | sync points (NCCL SendRecv 455 us + AllGather 5 x 22 us per decode step) | -0.4 decode ms (~0.9%) if they are not just absorbing skew | check what they wait on |

## Pruned groups, summary
PP/asymmetric placement (AR < weight time), decode DBO (memory-bound), tree drafts and drafter training (cost),
any lower precision of target/KV/AR/state (contract), smaller prefill chunks (fixed per-step cost).
