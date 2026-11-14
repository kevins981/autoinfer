# T2 sweep, step 4: the sweep, knob by knob

Reference: BASE calibrated (3-calibration.md): wall ~143 s = decode 2,500 x 38.2 ms (67%) + prefill-
containing 310 x 146.8 ms (32%). Gains below are napkin, as % of total wall time unless marked.
Two measurements made during this step (T2 GPU job 2, worker leader-2.w1, agents/leader-2.w1/out/):
- NCCL all-reduce, isolated, 2 ranks bf16 x5120: 16 rows 31.5 us, 48 rows 54.6, 260 rows 201, 2048
  rows 1467 (default protocol best or tied; LL/LL128 1.5-2x worse; channels no lever). Logits AG
  [45x124160] 891 us, [16x124160] 324 us. NCCL reaches ~13-14 GB/s, ~half the PCIe napkin.
  FlashInfer PCIe-IPC all-reduce: fails (needs peer access).
- Rank skew, from the BASE profile (both GPUs' AR kernels matched by end time): decode ARs: median
  skew 10 us, time after the later rank arrives 60 us (= real cost). **Prefill-step ARs: median skew
  150 us; after the later rank arrives only ~133 us; GPU0's AR kernels median 305 us.** So ~half of the
  62.5 ms "all-reduce" in prefill steps is one rank waiting for the other: the ranks are not in lockstep
  in prefill steps (eager/CPU-launch-bound parts, consistent with 24.5 ms GPU idle). nsys inflates CPU
  overhead, so the real size is unknown (to measure without nsys).

## Knob 1: Keep / Recompute (data nodes across steps)
| site | candidate | napkin | verdict |
|---|---|---|---|
| GDN state S (fp32, 1.57 MB/req/layer/GPU) | today: keep S after each of the 3 rows (write 3 slots) so any acceptance count can resume. **Row B (DeltaLog): keep S after the verified row only + the drafts' tiny inputs (k,v,q-norm'd, decay, beta: ~1 KB/head); next step replays the accepted drafts' updates in SRAM from S, same fp32 op order -> bitwise identical state.** | saves 2 of 3 state writes: 48 x 2 x 23.6 MB = 2.3 GB -> up to 2.3 ms/decode step; but measured GDN total is only 2.95 ms (calibration mismatch), so realistic 0.7-1.5 ms/step = **1.2-2.6%** | candidate (K1a); first check bytes with ncu |
| conv state | same trick | tiny | drop (bytes negligible) |
| KV cache | keep k,v (fp8) | — | already optimal; capacity not binding at 16 seqs |
| prefix (shared prompt prefix) | generic prefix caching | ShareGPT prompts share only the chat-template head | drop: ~0 for this workload (and benchmark-tuned caching is forbidden) |
| q, scores, activations | Rule 1/2 dead | — | drop |
| target hidden for MTP | already kept | — | — |

## Knob 2: Rewrite
| site | candidate | napkin | verdict |
|---|---|---|---|
| Sample = argmax_v over vocab-split logits | **combine tag argmax (max,idx) instead of concatenating all-gather**: each rank argmaxes its 124K-vocab half, gather 2 numbers/row. Exact for greedy rows (T=0); fall back to full gather for rows that sample or need logprobs | removes AGs 1.46 ms/decode step + rejection-sampler work on full logits (~0.1 ms) => 1.5 ms/step (3.9% of decode) + ~1.7 ms/prefill step => **~3.0%** | candidate (R1) |
| activations dtype of body GEMMs (contract allows checkpoint W8A8, per-token-group-128) | FP8xFP8 MMA (2x rate of bf16, no dequant) instead of Marlin | decode: if BW eff 0.66 -> ~0.85 TB/s: 18.5 -> 14.3 ms (save 4.2 ms/step = 7.3% of total); prefill: compute 38 ms bf16-floor -> 19 ms FP8-floor; Marlin 59 ms -> ~30 ms (save ~29 ms x 310 = 9 s = 6.3%) => **~10-13%** | candidate (R2); riskiest: a fast sm89 block-scaled FP8 GEMM at M=45 (Triton/CUTLASS), act-quant overhead, GSM8K A/B |
| Marlin itself at M=45 (same math) | tile/config tuning for M in 33-64, stages | 0.66 -> 0.75 TB/s: ~2 ms/decode step (3.5%) | candidate (R2b), cheaper test than R2 |
| residual-add + RMSNorm after each AR | fuse into the AR kernel (custom AR) | ~3.5 us x 128 = 0.45 ms/step (0.8%) | fold into C1 |
| GDN decode dual form for 3 verify rows | attention form (QKᵀ)V for draft rows from S(0) | same as K1a (avoids materialising S(1), S(2)) | merged with K1a |
| GDN a/b projections (bf16 small GEMMs) | concat with in_proj | ~0.2 ms/step; needs a precision change of those weights | drop (contract) |

## Knob 3: Speculate
| site | candidate | napkin | verdict |
|---|---|---|---|
| token edge t->t+1 (MTP, k=2, acceptance 2.345: per-position 0.78, 0.565) | k=3 | tokens/step 2.345 -> ~2.77 (+18%, assuming p3 ~0.42); cost +rows 45->60 (Marlin +2-3 ms, GDN +0.5-1 ms) + one more draft pass (~2.3 ms today) => step ~43.8 ms (+15%) => **+3% decode / +2% total**; with a cheap draft lm_head (S2) the extra pass is ~1 ms => **+5%** | candidate (S1), conditional on S2; TPOT p99 risk |
| draft lm_head (bf16 1.27 GB/GPU, read 2x per step; rulings allow FP8 or reduced vocab from non-benchmark text) | FP8 draft lm_head | 2 x (1.37 -> 0.7) = 1.34 ms/step => **2.3%** | candidate (S2a), maybe a config flag |
| | reduced-vocab draft lm_head (e.g. 32K most frequent tokens from a generic corpus) (+FP8) | 2 x (1.37+0.33 AG -> ~0.15) => ~3 ms/step => **5%**, minus acceptance loss on out-of-vocab tokens (napkin: <2% of tokens in English chat) | candidate (S2b); needs a non-benchmark token frequency source |
| tree speculation | top-B drafts per depth verified together | needs verification slack: at M=45 Marlin is already near its compute limit (bf16 ridge ~82 rows), each extra row costs; only with R2 (FP8 MMA, ridge ~165) | park, conditional on R2 |
| all-reduce outputs, layer outputs | high cardinality, p=0 (Gate 2) | — | drop |
| draft s -> s+1 | parallel drafting (both drafts in one pass) | needs a different draft model (MTP is trained sequentially) | drop (ML layer) |

## Knobs 4 and 5: Split / Merge and Place
| site | candidate | napkin | verdict |
|---|---|---|---|
| layer index l over 2 GPUs (PP=2) vs head/m split (TP=2) | PP=2 | decode memory-bound: PP makes each GPU stream 12.2 GB serially (or 2x per step with micro-batches): Marlin 18.5 -> ~37 ms vs saving 7.75 ms of AR => **loses ~11 ms/decode step**. Prefill (compute-bound): PP saves ~62 ms AR with micro-batching. Net: 2500 x -11 + 310 x +55 = -27 + 17 s => loses | drop; dominance proof: decode is BW-bound and 67% of time. Breaks only if AR cost stays >~ 140 us/op |
| asymmetric PP (33/31 + lm_head) | — | same as PP | drop |
| hybrid: PP for prefill, TP for decode | needs two weight layouts | 2 x 12.2 GB > capacity | drop (capacity) |
| r (DP) | full model per GPU | 24.4 GB weights > 24 GB | drop (capacity) |
| d split (sequence parallel: reduce-scatter + all-gather instead of AR) | same link bytes, lets norm run on half rows | ~0 on link; saves norm work only | drop |
| vocab v of lm_head | keep split + R1 combine | see R1 | — |
| vocab v of embedding | replicate embedding (1.27 GB more per GPU) to remove 3 embedding ARs/step | 3 x 60 us = 0.18 ms/step (0.3%) | drop (tiny; memory-picture ruling risk) |
| draft placement | whole draft on one GPU (no draft comm) with reduced-vocab lm_head | per draft pass ~0.95 ms vs ~2.35 today, but the other GPU idles | dominated by S2b at TP=2 (same lm_head saving, no idle GPU) |
| **all-reduce implementation (Place of the reduce through host memory)** | custom 2-GPU one-shot AR through pinned mapped host memory, chunked + flags, graph-capturable (no P2P needed) | NCCL ~50-60 us @45 rows, ~130-200 us @260 rows vs PCIe napkin ~25 / ~110 us. Decode: 130 x ~25-30 us = 3.5 ms/step (6% total); prefill bandwidth part: 128 x ~70 us = 9 ms/step (2%) => **~8%** | candidate (C1); test PCIe primitives first (job in progress) |
| KV / state offload | capacity not binding | — | drop |

## Knob 6: Window
| site | candidate | napkin | verdict |
|---|---|---|---|
| AR || next GEMM's weight reads | **prefetch the next GEMM's weights into the 72 MB L2 while the link-bound AR runs** (link || HBM; after AR#2 the next in_proj is 42 MB, after AR#1 gate_up is 89 MB: prefetch ~40 MB) | the next GEMM then reads ~40 MB from L2 instead of DRAM: up to ~40 us hidden per AR => 128 x ~40 us = 5 ms/decode step (**~9% total**); shrinks if C1 makes ARs shorter (with 25 us ARs: ~3 ms/step, 5%) | candidate (W2), novel; riskiest: L2 residency (evict policies, Marlin's own cache hints, NCCL's SM use) |
| AR || producing GEMM (split the row-parallel GEMM's output d in chunks; AR chunk c while computing c+1; no weight re-read since chunks read disjoint weight columns) | GEMM-comm overlap | hides min(GEMM, AR): down (57 us) + out/o (25-33 us) => ~4.8 ms/step if AR stays ~60 us | candidate (W1), overlaps with C1/W2 (same time pool) |
| split r into 2 half-batches, stagger (DBO) | — | shared-read rule: weights read twice at N=22 << ridge: +18 ms > 7.75 saved | drop (framework's DBO proof) |
| CPU gaps in decode steps (draft passes eager?) | full-graph the draft / remove host syncs | 1.03 ms idle/decode step => **1.8%** | candidate (W4a) |
| **prefill-containing steps not in lockstep (skew 150 us/AR, 24 ms idle under nsys)** | make mixed steps GPU-bound: CUDA-graph the mixed path (GDN prefill FLA kernels, attention), fewer launches | if the real (non-nsys) prefill step has ~20-40 ms of CPU-bound waiting: 310 x 30 ms = 9 s => **~6%** (range 0-10%) | candidate (W4b); first measure step times without nsys |
| merging more prefills per step | compute-bound at M=260 already | ~0 | drop |
| draft || target overlap | draft depends on the target's accepted tokens | — | drop |

## Summary of surviving candidates (napkin, % of total wall time)
R2 W8A8 GEMMs ~10-13% | W2 L2 prefetch during AR ~5-9% | C1 custom host-memory AR ~8% | W4b prefill-step
CPU/lockstep ~0-10% (unmeasured) | S2b reduced-vocab(+FP8) draft lm_head ~5% (S2a FP8 only ~2.3%) |
R1 argmax-combine logits ~3% | R2b Marlin tuning ~3.5% | S1 k=3 ~2-5% (after S2) | W4a decode CPU gaps
~1.8% | K1a GDN DeltaLog 1.2-2.6% | W1 GEMM-AR overlap (overlaps C1/W2).
Note: C1, W1 and W2 draw on the same pool (the ~7.75 ms/step of AR); they don't add.
