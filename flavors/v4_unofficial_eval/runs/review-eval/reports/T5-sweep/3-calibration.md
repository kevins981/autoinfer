# T5 sweep, step 3: napkin cost model, calibrated on the champion (T3-007)

Measurements: GPU job 77 (nsys TRACE=graph) and job 78 (TRACE=node), 64 ShareGPT prompts at c=16, champion
serve args, workspace t5-champ (= T3-007), files in agents/leader-5.w1/prof_graph, prof_node (worker
leader-5.w1). Whole-run counts: Prometheus of the champion's official trial 1
(bank/T3-007/eval/1/trial1/metrics_after.txt) and its bench_c16.json.
Hardware constants: BW 1.0 TB/s, F 165 TFLOP/s (FP8, fp32 acc), PCIe ~25 GB/s/direction.

## A. Whole benchmark: two step types
Official trial 1: 106.6 s, 2,739 engine steps (`iteration_tokens_total_count`), of which 2,368 have
<= 64 tokens (decode, or decode + a tiny prefill) and 371 have > 64 tokens (prefill-containing, "mixed").
| | steps | period (measured) | time | share |
|---|---|---|---|---|
| decode (M~45-48) | 2,368 | 28.2 ms (graph trace median; 31.3 ms at full batch D>=15) | 66.8 s | 63% |
| mixed (1 prefill of ~280 tokens avg + 45 decode rows) | 371 | ~107 ms (= (106.6 - 66.8) / 371; nsys median 113.6) | 39.8 s | 37% |
The profile's own split agrees (decode 65%, prefill-containing 35% of the captured window).

## B. Decode step (per GPU), napkin vs measured (node trace, median of 246 full-batch steps, 31.6 ms wall)
| Part | Napkin | Measured | Ratio | Note |
|---|---|---|---|---|
| Body GEMMs (12.17 GB FP8 weights) | 12.2 ms | 15.3 ms (gate_up 7.25, down 3.47, qkvz 1.96, out/o 1.59, qkv 0.56, ba etc 0.46) | 0.80 of BW | gate_up 89 MB in 118 us = 0.75 TB/s; down 0.81; qkvz 0.66 |
| All-reduce, 128 body (0.48 MB each) | 128 x (19 us link + ~20 us latency) = 5.0 ms | 6.2 ms (hmar ~47 us each) | 0.81 | serial; L2 prefetch (4.7 ms of side-stream kernels) overlaps it |
| GDN recurrent state (fp32, read 1 + write 3 slots per req/head) | 1.21 + 3.63 GB = 4.8 ms | 3.69 ms (conv + gdn_decode_post_conv_mtp, ~72 us/layer) | 1.3 (faster than napkin) | MISMATCH: writes cost less than the napkin; either part of the 3 writes is skipped (dest slot 0 = no write: padded/finished rows) or absorbed by L2 write-back. Read-only floor 1.2 ms. |
| Target lm_head (bf16 1.27 GB) | 1.27 ms | 1.58 ms | 0.80 | |
| Drafter, 2 steps (0.64 GB fp8 head + ~0.21 GB layer each, 2 AR) | 1.7 + 0.2 = 1.9 ms | 2.85 ms | 0.67 | head 875 us/step (0.73 TB/s) |
| Attention (KV ~0.16 GB) | 0.16 ms | 0.50 ms | | small |
| Norms/act/quant, sampling (SendRecv 0.53), input prep, other | ~0.5 ms | ~1.9 ms | | |
| Launch/idle gaps | 0 (FULL graph) | 0.15 ms | | not a lever for decode |
| **Total** | **25.6 ms** | **31.6 ms node / 28.2-31.3 graph** | **0.81-0.91** | ranking of parts preserved |

## C. Mixed step: the napkin needs a resource the framework does not have (the host CPU)
GPU napkin for a step with p prefill rows + 45 decode rows: max(weights 13.5 ms, (45+p) x 25.6 GFLOP / 165 TF)
+ 130 ARs (NCCL above 96 rows: ~105 us each measured at 107 rows = 13.7 ms) + GDN chunk kernels (~5 ms).
For p = 60: ~16 + 13.7 + 5 = ~35 ms; measured GPU busy ~50 ms (job 77 per-step rows: p=57: 49.6, p=110: 61.5,
p=199: 84.5, p=358: 115.9, p=708: 178.5 ms). Fit: GPU busy ~ 37 ms + 0.21 ms per prefill token
(0.21 ms/token = 25.6 GFLOP / 120 TFLOP/s effective: compute-bound, 73% of FP8 peak).
**Measured period is ~110-114 ms flat from p=57 to p=358**, then follows the GPU. In a p=59 step the TP0 worker
thread makes 2,915 CUDA API calls (1,093 cudaLaunchKernel, 560 cuLaunchKernelEx = Triton, 65 cudaGraphLaunch)
with no gap > 1 ms: the step is host-bound. Node trace of a p=62 step: ~1,650 eager kernels, nearly all in the
47 GDN layers' mixed path (index_select/gather 285+201, elementwise 335+157+106, the ~10 FLA chunk kernels x 47,
fused_sigmoid_gating spec update x 47, conv1d fwd+update, gated layer norm), i.e. ~30 eager launches per GDN layer.
Extended napkin: period = max(CPU, GPU), CPU = N_host_ops x t_op with t_op ~ 37 us (2,915 calls in 110 ms,
incl. the Python between them). It reproduces the flat ~110 ms floor and the crossover at p ~ 350.
Real (non-nsys) timing agrees: section A's 107 ms average real mixed period ~ nsys's 113.6 ms median.

## D. What the calibrated model says, by size of the lever (against the champion's 106.6 s)
| Lever | Time in the champion | Floor | Room |
|---|---|---|---|
| Mixed-step host overhead (CPU-bound floor ~110 ms vs GPU 50-90 ms for p < 350) | ~39.8 s total mixed time; ~10-18 s of it is GPU-idle host time (estimate: sum over steps of 110 - GPU busy) | GPU busy | 10-17% |
| Decode body GEMM efficiency (0.80 of BW) | 15.3 ms/step x 2,368 = 36 s | 12.2 ms | up to ~3 ms/step (~7%), hard: tile configs already tuned |
| Decode all-reduce latency | 6.2 ms/step = 14.7 s | link-bound ~2.4 ms | protocol-level, hard |
| GDN state writes | 3.7 ms/step = 8.7 s | ~1.2-1.8 ms | ~2 ms/step (~4%), needs state-slot redesign |
| Drafter + lm_heads | 2.85 + 1.58 ms/step | | reduced-vocab draft head ~1.5 ms/step (~3%), acceptance risk |
| Mixed-step ARs via NCCL (rows > 96) | 130 x ~105 us = 13.7 ms per mixed step (GPU side) | hmar-like ~50-70 us | only matters once mixed steps are GPU-bound (p > 350 now) |
