# T6 sweep, step 4: knob by knob over the graph (gains vs T3-007's measured time)

Reference times (3-calibration.md): decode step ~25 ms (~72% of wall), prefill/mixed step ~100 ms (~28% of wall); total 106.6 s.
Gain % = wall-time saved / 106.6 s (throughput gain ~ the same for small values).

## Knob 1: Keep / Recompute (data nodes across steps t)
| site | options | napkin | verdict |
|---|---|---|---|
| **GDN state S[r,l] under MTP (3 rows/step)** | A (now): keep S after every row (1 read + 3 fp32 writes per request-layer); **B (ReplaySSM, 02 §4.3): keep S_ckpt + the step's per-row (k, v, decay, beta) (~1 KB/head/row); next step replays the `accepted` rows from S_ckpt, then writes 1 state**; C: keep k,v history only (no S) | A: 3.8 GB/step traffic; B: 1.9 GB/step + ~0.02 GB -> **-1.2..1.9 ms per decode step (kernel time vs byte napkin)** -> **+3.5..5%**. Also frees 2/3 of the spec state slots (~2.4 GB/GPU). Exact: the replay repeats the same fp32 ops in the same order (bitwise identical). | **Candidate 2.** Risk: mixed steps use a second (Triton) kernel and mamba "align" prefix-cache mode manages state slots; both must keep the contract |
| conv state (3 rows of conv input) | same pattern | tiny bytes | not worth separately |
| KV cache k,v (fp8) | keep (fixed by contract) | — | fixed |
| q, s, alpha, logits | Rule 1/2 dead | — | dead |
| prefix cache | on (contract) | — | fixed |

## Knob 2: Rewrite
| site | napkin | verdict |
|---|---|---|
| target lm_head (bf16, 1.27 GB/GPU, N=48) | must stay baseline precision; argmax-only already via batch-sharded sampling | no byte reduction possible: dead |
| draft lm_head (FP8, 0.64 GB/GPU, read 2x/step) | reduced vocab (ruling 2, non-benchmark frequencies) 248K -> 32K: -1.1 ms/decode step = +3% minus acceptance loss (approx in the draft only; verification exact) | candidate 5; needs a generic corpus for frequencies |
| merge in_proj_ba (bf16) into qkvz | ba is tiny (N=96 cols); saves a launch | < 0.3%: dropped |
| GDN chunk form vs recurrent form in verify (03 §4.2: M=3 < d=128 -> attention form) | the recurrent kernel's cost is state traffic, not FLOPs; the attention form needs the state anyway | no gain: dropped |

## Knob 3: Speculate
| site | napkin | verdict |
|---|---|---|
| next token, chain length k=2 -> 3 | tokens/step 2.34 -> ~2.74 (pos3 ~0.40 extrapolated from 0.78, 0.56): +17%; cost: target M 48->64 (still < N_ridge 82: ~+5% GEMM), +1 draft step (~1.5 ms), +1 GDN state write (~+1 ms; 0 with K1-B) -> step +3.5 ms (+14%) | **+3% alone, ~+7% with K1-B.** Candidate 4 |
| tree (top-2 at depth 1) | M up to ~80 < ridge; but GDN verification of branches needs per-branch states (no shared append-only history: 02 §4.4 trap) | implementation too large for this run: dropped (noted) |
| router/expert | dense model | n/a |

## Knob 4: Split / Merge (indices across the 2 GPUs or within a device)
| site | napkin | verdict |
|---|---|---|
| TP=2 (now) vs PP=2 | decode N=48 < 82: PP with 2 microbatches re-reads weights (2 x 6.1 GB per GPU per step vs 6.1 + AR 5.4 ms) -> loses at decode; 1 microbatch idles a GPU | dead (dominance: weights re-read > AR time at N < ridge) |
| DP for attention/GDN + TP for MLP | needs replicated attention/GDN weights (~4 GB more per GPU) + KV/state per GPU; AR count halves (MLP only) but adds all-gathers | capacity tight; complexity high: dropped |
| AR -> reduce-scatter + all-gather (sequence-parallel norms) | link bytes equal at 2 ranks | dropped |
| **split M of prefill steps into 2 micro-batches (Window + Split r)** | prefill GEMMs compute-bound (N=130 > 82 still) -> weights re-read is free; AR (link) overlaps the other half's GEMM (compute): 119 -> ~max(55,57)+tail ~ 70 ms | **+~10% at the current AR cost; much less once the AR is fixed.** Candidate 3 (dual-batch overlap, 06 §3) |

## Knob 5: Place
| site | verdict |
|---|---|
| draft MTP layer on one GPU (no TP) | saves 4 ARs/step (~0.18 ms) but the full MTP weights + draft lm_head on one GPU (0.4 + 1.27 GB) read by one GPU: loses |
| **all-reduce transport for large messages (prefill M~260, 2.6 MB)**: NCCL RING_LL via host SHM (now, 421 us) vs host-mapped pinned buffer written/read directly by the GPUs (the champion's HMAR, today capped at 96 rows / 1 MB) | napkin 2.6 MB / ~17 GB/s (HMAR's measured rate at 480 KB) ~ 160-200 us -> -30 ms per prefill step -> **+6..9%** | **Candidate 1** (cheapest; same exact single fp32 add per element) |
| KV / state offload to host | capacity is not binding | dead |

## Knob 6: Window
| site | verdict |
|---|---|
| AR || L2 prefetch of next weights (done by T3-001) | in champion |
| decode: GDN recurrent (DRAM) || AR (link) | data dependence (AR output feeds the norm before GDN): no |
| prefill dual-batch overlap | see Knob 4 (candidate 3) |
| prefill-step GPU idle 28 ms (CPU-side work between kernels) | unexplained; needs a graph-level trace: candidate 6 (investigate) |
| decode GEMM efficiency (76% of DRAM peak, ~3.9 ms/step) | Implementation layer; earlier teams tuned tiles: candidate 7 (low expected) |

## Summary of candidates (napkin)
1. Large-message all-reduce through host-mapped memory: +6..9%.
2. ReplaySSM for GDN spec states: +3.5..5%.
3. Prefill dual-batch overlap: +~10% now, small after #1.
4. 3 draft tokens: +3% (+7% with #2).
5. Reduced-vocab draft lm_head: up to +3% minus acceptance loss.
6. Prefill-step idle: unknown.
7. Decode GEMM efficiency: up to +4%, hard.
