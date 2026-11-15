# T4 sweep, step 4: knob by knob over the graph

Baseline for gains: HEAD T3-004, calibrated in 3-calibration.md. Whole run ~99.5 s; shares:
decode-only steps ~62% (step ~25 ms), mixed/prefill extra ~38%. Gains below are % of whole-run
time (≈ % throughput) unless marked. GPU-0 whole-capture totals (19.34 s capture): W8A16 7.77 s,
W8A8 2.88 s, NCCL AR 2.43 s, host AR 2.41 s, lm/draft heads etc 1.88 s, GDN 1.15 s.

Contract filter applied to every row: exact up to fp reordering (or the allowed W8A8 scheme with
its GSM8K A/B); comm at bf16; FP8 KV; GDN state fp32; exact greedy verify; prefix caching on.

## Knob 1: Keep / Recompute (data nodes across steps)
| site | candidate rows | napkin | verdict |
|---|---|---|---|
| k,v (ATT) | keep FP8 (fixed) | — | contract |
| GDN state S | A: in-place every token; B: replay pending rows, commit every 2nd step (HEAD); B': commit every 3rd step | B->B': writes 0.6 -> 0.4 GB/step = -0.2 ms (0.8% decode, ~0.5% total) but replay rows 4->6, shared-mem / kernel complexity | low; keep listed (K1-lazy3) |
| GDN state precision | bf16 state (approx) | -0.9 GB/step | **forbidden** (lossy: state dtype is the model's) |
| conv state, logits, q | no future readers / tiny | 0 | dead (Rule 1) |
| prefix KV/state | default prefix caching | benchmark has ~no shared prefixes | no tuning allowed |

## Knob 2: Rewrite
| site | rewrite | napkin | verdict |
|---|---|---|---|
| decode GEMM x·W (M=48), 15.4 ms/step at 0.79 TB/s | (a) W8A8 (checkpoint scheme) so the mainloop has no fp8->bf16 dequant; (b) same math W8A16, better kernel | roof 12.2 ms @1 TB/s; at 0.92 TB/s: 13.2 ms => **-2.2 ms/step = 8.8% decode ≈ 5.5% total** | (b) exact, preferred; (a) needs GSM8K A/B. Risk: HEAD's Triton tables are already tuned; Triton W8A8 at small M is slow (gate_up M=99: 0.54 TB/s). Implementation-level; candidate **R1** |
| AR -> residual add -> RMSNorm | fuse (one kernel reads peer data, adds, normalizes) | 130 x ~2.8 us = 0.36 ms/step (1.4% decode) + fewer launches | small; **R2**, natural add-on to a new AR kernel |
| target lm_head 1.27 GB bf16, 0.80 TB/s | better GEMV/GEMM kernel | -0.2 ms/step | small (R3) |
| MTP fc on [emb; h] | split into emb·W1 + h·W2 | no saving (both computed anyway) | dead |
| lm_head argmax | vocab-parallel argmax merge | already in HEAD | done |
| prefill W8A8 FP8 mma, fp32 acc (330 TF) vs fp16 acc within K-block (660 TF) | accumulate each 32/64-K sub-block in the tensor core's fp16 accumulator, promote to fp32 | W8A8 2.88 s -> ~1.6 s: **~6.5% total** | approx-ish: overflow (e4m3 products up to 2e5 > fp16 max) and precision; TASK says "the FP8 tensor cores' own accumulator is fine within a block" — whether fp16 acc counts needs a **ruling**; listed as R4 (ask) |
| GDN verify rows (M=3) | chunk/dual form | already one fused kernel | done |

## Knob 3: Speculate
| site | candidate | napkin | verdict |
|---|---|---|---|
| next token (MTP), k=2 -> k=3 chain | +1 draft step | tokens/step 2.33 -> ~2.73 (assume 3rd conditional acceptance 0.70, from pos1 0.78, pos2|1 0.73); step cost +0.65 ms draft, +~0.5 ms GEMM tiles at M=64, +0.9 ms AR (0.66 MB msgs), GDN replay must handle 3 drafts (else +1.3 ms state writes) => 25 -> ~27.2 ms: **+7.6% decode tok/s ≈ +4.5% total**; if replay can't extend: ~+2% | **S1**; riskiest assumption: 3rd-position acceptance (MTP head trained for 1 step); cheap test: run k=3 on a short bench and read per-position acceptance |
| tree (top-2 at depth 1) | branches need separate GDN state paths | +~0.08-0.1 tokens/step for +16 rows | dropped: GDN state branching cost (02 §4.4 trap) and complexity vs ~3% |
| draft head vocab 98K -> smaller | 2 x 0.17 GB less | -0.35 ms/step (1.4% decode) minus acceptance loss | low (S2) |
| layer output / AR result / activations | high cardinality, p≈0 | — | dead (Gate 2) |
| stage-speculative PP | GPU0 starts step t+1 on assumed-all-accepted drafts | P(all accepted) ~0.57: 43% wasted | dropped: needs PP (below) and low p |

## Knobs 4-5: Split / Merge and Place
| site | candidate | napkin | verdict |
|---|---|---|---|
| TP=2 vs PP=2 (2 micro-batches of 8 req) | PP removes 128 ARs (3.1 ms) but every GPU reads its 12.17 GB once per micro-batch = 24.3 GB per full step | +12 ms vs -3 ms | **dead** (reuse N halves; memory-bound) |
| PP=2, 1 micro-batch | one GPU idle half the time | 2x | dead |
| asymmetric TP split (alpha != 0.5) | ranks are symmetric (same layers, same lm_head shard) | 0 | dead (05 §4.1 condition not met) |
| **transfer node of every AR (link)** | the link is two resources: PCIe up (GPU->host) and down (host->GPU). HEAD's one-shot host AR does write-all then read-all (2·S/BW); NCCL LL over SHM gets 8-15 GB/s for 1.4-9 MB. Split the message into sub-chunks and Window them: write chunk k+1 while reading chunk k | decode: 30 -> ~22 us x 130 = -1.0 ms/step (4% decode, 2.5% total); prefill: NCCL 2.43 s -> S/25 GB/s ≈ 1.4 s => -1.0 s (5% total) | **P3, top**: one kernel, exact (same bf16 a+b), CUDA-graph capturable |
| split d (output columns) of row-parallel GEMMs + Window with the AR (decode) | chunk k's AR overlaps chunk k+1's GEMM, weights read once | ~-8 us per AR x 130 = -1.0 ms/step (2.5% total); tile-level fusion (GEMM epilogue publishes tiles, next norm waits per tile) could hide ~2.5 ms/step (6% total) | **P2**; needs concurrent kernels in graphs; after P3 |
| split r/i into 2 micro-batches in mixed steps (DBO, staggered windows) | AR of one half under GEMMs of the other; halves of ~170 rows are near the W8A8 ridge (~165) so the extra weight read costs little | hides most remaining prefill comm: after P3 ~1.4 s -> ~0.4 s: **~5% total** | **P4**; complex (two interleaved forwards, metadata split); HEAD's row-chunked GEMM+AR does part of it, eager only, only within one GEMM |
| expert / context parallel | no MoE; contexts short | — | n/a |
| place KV/state on host or peer | capacity not binding at c=16 | — | dead |

## Knob 6: Window (merge and overlap)
| site | candidate | napkin | verdict |
|---|---|---|---|
| decode rows + prompt rows in one window | already merged (chunked prefill) | — | done |
| several prompts per mixed step (hold prefills) | mixed steps are compute-bound: saves only per-step fixed costs (~decode-part) | ~1-2% | parameter choice must not be tuned on the benchmark; low (W1) |
| GDN kernel (state BW) || GEMM (weight BW) | same resource | 0 | dead |
| draft s=1 || s=2 | serial | — | dead |
| host-side gaps (GPU busy 95.8% of wall) | async scheduling / fewer syncs | up to ~2-4% | unpriced mechanism; profile with TRACE=graph first (W2) |

## Groups dropped and why (summary)
- PP / asymmetric splits: memory-bound decode, symmetric ranks.
- Tree speculation, stage-speculative PP: GDN state branching and low p.
- Lossy rows (bf16 state, FP8 comm, smaller weights): contract.
- Layer/activation speculation: Gate 2 (cardinality).

## Ranked short list (napkin, % of total)
1. P3 pipelined host-memory AR for all sizes (decode + prefill): ~7%.
2. R1 decode GEMM at ~92% of DRAM BW (exact kernel): ~5.5%.
3. P4 prefill micro-batch overlap of AR: ~5% (after P3).
4. S1 MTP k=3 with 3-draft replay: ~4.5% (acceptance-dependent).
5. P2 AR/GEMM tile overlap in decode: 2.5-6%.
6. R4 fp16-accumulate prefill (needs ruling): ~6.5%.
7. small: R2 AR+norm fusion 1%, S2, K1-lazy3, W1, W2.
