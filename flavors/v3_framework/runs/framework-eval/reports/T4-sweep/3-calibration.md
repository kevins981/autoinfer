# T4 sweep, step 3: napkin cost model, calibrated against the champion

Measurements: champion T3-009 (ws t4-champ), nsys under 64 ShareGPT prompts at c=16, GPU jobs 139 (TRACE=graph:
agents/leader-4/prof_graph) and 140 (TRACE=node: agents/leader-4/prof_node; worker summary agents/leader-4/prof_summary.md).
Prompt lengths of the actual 400 benchmark prompts (chat template applied, same sampler and seed as the harness, CPU only):
agents/leader-4/plen.json (`plen.py`): sum 104,526 tokens (official total_token_throughput implies ~105K ✓), mean 261,
p25 71, median 128, p75 396, p90 681, p99 885; **232 prompts < 211 tokens, 92 in [211, 467), 76 >= 467** (P + ~45 decode
rows > 512 = the largest CUDA-graph capture size).

## 1. Whole-benchmark decomposition (the level the objective lives at)
Model: wall = N_dec * T_dec + Σ_prompts T_mixed(P). Each prompt causes ~one mixed step (one prefill per step; 16 slots,
closed loop). Mixed steps also advance ~15 decoding requests (~35 tokens).
- T_dec = 26.45 ms (graph trace, median of 497 decode steps; node trace 26.67).
- N_dec = (91,293 output tokens - 400 * 35) / (16 * 2.33) = 2,070 steps -> **54.9 s**.
- T_mixed(P), measured per bucket (graph trace, 50 mixed steps): P+45 < 256: ~75 ms (57-145 prompt tokens all ~70-78 ms,
  a few ~110 ms); 256-512: ~80-100 ms (~90); > 512 (eager, no graph): ~205-220 ms for P 650-860, ~350 ms for multi-prompt
  steps of ~1900 tokens.
- Σ T_mixed over the 400 real prompt lengths: 232 x 75 + 92 x 90 + 76 x ~215 = 17.4 + 8.3 + 16.3 = **42.0 s**.
- **Napkin total 96.9 s vs measured 95.9 s** (official: 91,293 tokens / 951.5 tok/s). Within 1%. Cross-check from official
  latencies: median TPOT 15.5 ms x 2.33 = 36 ms average step: 26.5 (1-f) + ~105 f = 36 -> f ~ 12% of steps mixed, ~2,500
  steps total: consistent with ~2,070 decode + ~400 mixed.

| Part | Measured share of wall | Napkin floor (below) | Gap |
|---|---|---|---|
| Decode-only steps | 54.9 s (57%) | 2,070 x ~17-19 ms = 35-39 s | ~16-20 s |
| Mixed steps, P < 211 | 17.4 s (18%) | 6.6 s | ~10.8 s |
| Mixed steps, 211 <= P < 467 | 8.3 s (9%) | 6.3 s | ~2 s |
| Mixed steps, P >= 467 (eager) | 16.3 s (17%) | 9.4 s | ~7 s |

Mixed-step napkin floor: max(T_dec, (P+45) rows x 24.3 GFLOP/row / 150 TFLOP/s + 5 ms (GDN state, lm_head, draft)).
24.3 GFLOP/row/GPU = 2 x 12.17e9 FP8 params per GPU. 150 TFLOP/s is the measured Triton W8A8 rate in mixed steps
(node trace: 44.9 ms of W8A8 GEMM for 276 rows = 6.7 TFLOP -> 150 TF/s, 91% of the ~165 TF/s fp32-accumulate spec): mixed
steps are compute-bound in their GEMMs, as the ridge (~80 rows for FP8 at fp32 accumulate) predicts.

## 2. Decode step (M = 48 rows), napkin vs measured, per GPU (node trace, GPU0, us per step)
| Part | Napkin (bytes / 1.0 TB/s, or link) | Measured | Ratio | Note |
|---|---|---|---|---|
| Body W8A8 GEMMs (256 + 8 MTP) | 12.17 + 0.37 GB = 12.5 ms | 14.43 ms | 1.15 | ~0.87 TB/s effective; gate_up 89 MB in 106 us = 0.84 TB/s, in_proj 42 MB in 40 us = 1.05 TB/s (L2 prefetch helps), down 44.6 MB in 52 us, out/o 15.7 MB in ~21 us = 0.75 TB/s |
| All-reduces (128 + MTP 4 + gathers) | link: 491 KB each way / ~25 GB/s ~ 20 us + flag latency -> ~2.6-3 ms if not hidden | 6.91 ms (135 calls, ~52 us each) | 2.3 | the AR kernel also prefetches 40 MB of the next GEMM's weights into L2 (64 extra CTAs), so its duration ~ max(link, 40 MB at HBM rate) |
| GEMM + AR together | weights at 1.0 TB/s, AR fully hidden: 12.5 ms | 21.3 ms | 1.7 | **the biggest decode gap: ~8 ms** |
| GDN decode kernel (48 x) | state read 25 MB + write 25 MB (row-0 full state, rows 1-2 logs) per layer = 50 MB -> 50 us | 37 us each, 1.78 ms | 0.74 | faster than napkin: partly L2 hits (writes of 25 MB/layer fit in 72 MB L2; read-after-write next step evicted though). DeltaLog materialize: +87 us/step |
| Target lm_head bf16 | 1.27 GB -> 1.27 ms | 1.58 ms | 1.25 | cuBLAS ampere bf16 at 0.80 TB/s |
| Draft lm_head FP8 x2 | 2 x 0.39 GB = 0.78 ms | 1.03 ms | 1.3 | cutlass FP8 with visitor, 0.76 TB/s |
| MTP rest (layer, norms, ARs) | ~0.4 ms | ~1.2 ms | | whole draft part ~2.2 ms |
| Norm/quant, conv, attention, misc | ~0.5 ms | ~1.3 ms | | small kernels, ~1-5 us each, ~600 launches inside graphs |
| **Step** | **~17 ms (link hidden) / ~19.6 (not hidden)** | **26.5 ms** (graph trace period; GPU 99% busy) | 1.35-1.55 | |

Interpretation: the napkin reproduces the parts, and the residual is concentrated in (a) GEMM+AR (8 ms over the
bytes-only floor), (b) memory-bound kernels running at 0.75-0.87 TB/s rather than ~0.95, (c) launch-sized small kernels.
The decode step is not launch-bound (GPU idle 0.16 ms/step, one full graph per step plus the eager lm_head).

## 3. Mixed step anatomy (why P = 60 costs as much as P = 300)
Graph trace, step `execute_context_1(65)_generation_15(45)` (110 rows), GPU0: per GDN layer one graph segment of
~0.545 ms (GEMMs + norms + AR) then the GDN core runs **eagerly** (breakable CUDA graph: GDN attention is a break point
outside uniform-decode steps): ~30 small kernels (index_select/gather, conv update + conv fwd, cat, post-conv prep,
DeltaLog materialize, Triton fused_sigmoid_gating_delta_rule_update for the 15 spec-decode requests (74 us), the 6-kernel
Triton chunked-prefill pipeline for the prompt, index_copy merges) with **~1.4 ms wall per layer but ~0.2 ms of GPU work**:
the GPU waits for Python and Triton launches. 48 layers x ~1.2 ms idle = ~55 ms of the ~75-105 ms step.
- For P + 45 >= 256, the GPU work per layer grows past the CPU time, so the same CPU cost hides (steps ~80-100 ms,
  GPU-bound: 45 ms W8A8 GEMM, 7 ms GDN, AR partly overlapped by row-split; but the row halves are 1.4 MB > the hostmem
  AR's 1.31 MB limit, so they go to NCCL ring at ~217 us per call, 36 ms/step of NCCL partly hidden).
- For P + 45 > 512 there is no captured graph at all: everything eager, ~210+ ms (GPU busy ~150-180 ms).

## 4. Constants this machine actually gives (used in step 4)
| Constant | Value used | Source |
|---|---|---|
| Effective HBM BW for big GEMMs / GEMV | 0.85-0.9 TB/s now; ~0.95 achievable (assumption, check) | node trace |
| FP8 W8A8 compute (fp32 acc across 128-blocks) | 150 TF/s achieved, 165 spec | node trace |
| Ridge (FP8, fp32 acc) | ~80-160 rows | spec / mixed steps compute-bound at 110+ rows: 0.545 ms/layer for 110 rows vs 0.375 for 48 |
| hostmem AR, 491 KB | 52 us incl. 40 MB L2 prefetch | node trace |
| NCCL ring AR, 1.4 MB | 217 us | node trace |
| Eager GDN mixed core | ~1.4 ms CPU per layer, ~0.2 ms GPU | graph trace |
| Per-kernel launch inside graphs | ~1-2 us gap | node trace |
