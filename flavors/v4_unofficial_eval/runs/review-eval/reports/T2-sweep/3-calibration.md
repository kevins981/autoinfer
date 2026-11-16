# T2 sweep, step 3: napkin cost model, calibrated against BASE

Measured data: task/baseline/nsys_c16 (64 prompts at c=16, node-level graph trace, GPU 0 unless
noted; per-step numbers from `task/scripts/step_anatomy.py` and my per-kernel script
`agents/leader-2/stepkern.py` / `seq2.py`, decode steps with exactly 48 rows: 186 steps) and
task/baseline/official (400 prompts: 143.2 s, 91,080 output tokens, 104,526 input tokens,
38,713 drafts, acceptance length 2.345, ITL p50/p90/p99 39.3/117/268 ms).
Caveat: nsys inflates eager (non-graph) launches; decode steps are graph-replayed (little distortion),
mixed steps are partly eager (inflated).

## Hardware constants used (and the correction calibration forced)
- DRAM 1.0 TB/s. Link: PCIe via host, 25 GB/s/dir nominal (worker microbenchmark pending: leader-2.w1).
- bf16 tensor with fp32 accumulate: I first assumed 82.6 TFLOP/s. **Wrong**: in large mixed steps
  Marlin does 894 tokens x 24.4 GFLOP = 21.8 TFLOP in 172 ms = **127 TFLOP/s achieved**, so the
  bf16 fp32-acc peak must be >= 165 (Ada whitepaper: 165 dense). Corrected: bf16 165, so the
  Marlin ridge is ~82 rows, and decode (M = 48) is memory-bound, not compute-bound.
  FP8 fp32-acc: 165 (SPEED_OF_LIGHT) or 330 (my reading of the whitepaper): **open, the worker's
  GEMM microbenchmark decides**; it sets the value of W8A8 for prefill.

## A. Decode step, 16 requests x 3 rows (M = 48), per GPU
| part | napkin (ideal) | measured (nsys, median) | ratio meas/napkin | notes |
|---|---|---|---|---|
| body FP8 GEMMs (Marlin) | 12.2 GB / 1.0 TB/s = 12.2 ms (compute 1.17 TF/165 = 7.1 ms) | 18.57 ms | 1.52 | ncu in isolation: 630-800 GB/s (gate_up 115 us vs 141 us in-server) |
| all-reduce, 135 x 0.49 MB | 135 x (0.49 MB / 25 GB/s + ~10 us) = 4.0 ms | 8.16 ms (60 us each) | 2.0 | RING_LL, 2 channels, SHM |
| logits all-gather: target 12 MB + 2 x draft 4 MB | 20 MB / 25 GB/s = 0.8 ms | 1.67 ms (0.95 + 2 x 0.33) | 2.1 | **needed only because argmax is done after gathering** |
| target lm_head, 0.635 GB bf16 | 0.64 ms | 1.59 ms (ampere_bf16 128x64, 970 CTAs) | 2.5 | |
| draft lm_head x 2, M = 16 | 2 x 0.64 = 1.27 ms | 2.74 ms (cutlass wmma 16x16, **grid 8 CTAs**) | 2.2 | kernel uses 8 of 128 SMs |
| GDN state read+write fp32 (48 layers) | 2.4 GB / 1 TB/s = 2.4 ms | 3.66 ms (gdn_decode_post_conv_mtp) | 1.5 | |
| attention (16 layers) + KV write | ~0.1 GB -> 0.1 ms | 0.50 ms | (latency-bound) | |
| MTP layer x 2 (FP8 ~0.2 GB) | 2 x 0.2 ms | ~2 x 0.45 ms (Marlin 256,1,8,8 + AR) | ~2 | |
| norms, act, conv, rope, small | ~0.3 ms | ~1.3 ms | | many tiny kernels |
| sampling / rejection / bookkeeping | ~0 | ~0.25 ms (_resample 70 us, logits stats 49 us, elementwise on logits 45 us) | | |
| GPU idle inside step (CPU prep between graphs) | 0 | 1.3 ms (step_anatomy) | | gaps of 120-250 us before draft steps |
| **total** | **~22.5 ms** | **38.5 ms** (= ITL p50 39.3 ms in the official run) | 1.71 | |

Sum of measured parts: 18.6 + 8.2 + 1.7 + 1.6 + 2.7 + 3.7 + 0.5 + 0.9 + 1.3 + 0.3 + 1.3 = 40.8 ms
(the parts overlap the 38.5 ms period by ~2 ms: the MTP layer's Marlin is inside "body" partly;
good enough). Calibrated model = napkin x per-part ratio above; it reproduces the decode step
within ~5%.

## B. Mixed (prefill-containing) step: n prompt tokens + ~45 decode rows
From the 50 mixed steps of the profile (GPU 0 / GPU 1):
| n+45 tokens | period | Marlin | NCCL | GDN | GPU idle (GPU1) |
|---|---|---|---|---|---|
| 102-115 | 120-128 ms | 33-38 ms | 58-71 (GPU0) / 15-16 (GPU1) | 5.8 | 54-65 ms |
| 244 | 128 | 56 / 49 | 47 / 32 | 7.4 | 31 |
| 375-447 | 149-166 | 82-91 / 71-78 | 42-48 / 52-61 | 8.7 | 6 |
| 651-894 | 225-296 | 129-172 / 109-146 | 69-90 / 87-118 | 10-12 | 3-8 |
| 1885 | 589 | 354 / 298 | 180 / 243 | 17 | 6 |

Napkin per token (large n, compute-bound): Marlin 24.4 GFLOP/token / 165 TF = 0.148 ms/token,
measured 0.16-0.19 ms/token (ratio ~1.2); all-reduce 135 x 10.2 KB = 1.38 MB/token / 25 GB/s =
0.055 ms/token, measured 0.10-0.13 ms/token (ratio ~2: ~11-14 GB/s effective).
Small mixed steps (n < ~250) are **not GPU-bound**: one GPU idles ~60 ms of a ~125 ms step while
the other's NCCL kernel waits for it. Cause: the mixed step exceeds the largest captured graph
(96 tokens), so it runs eagerly and the 48 GDN layers' chunked-prefill path is launched kernel by
kernel from Python (~2.5 ms per GDN layer under nsys, mostly CPU gaps of 20-170 us between
1-10 us kernels). nsys inflates this; the official run's ITL p90 of 117 ms says small mixed steps
are still ~100+ ms without the profiler.

Calibrated mixed-step model: T_mix(n) ~= 40 ms (decode part) + 0.30 ms x n (Marlin 0.19 + NCCL 0.10)
+ E, with E = eager/CPU overhead.

## C. Whole benchmark (official run, 143.2 s)
- Steps: 38,713 drafts / ~15.5 running ~= 2,500 steps; ~370 of them mixed (one new request each,
  first steps admit several).
- Decode steps: 2,500 - 370 = ~2,130 x 39.3 ms = **~84 s (59%)**.
- Mixed steps: 143 - 84 = **~59 s (41%)** = 370 x 40 ms (decode part, 14.8 s) + 104.5K prompt tokens x
  0.30 ms (31 s) + 370 x E  ->  **E ~= 35 ms per mixed step (~13 s total)**.
- Check against SPEED_OF_LIGHT: optimistic floor 45-53 s; BASE is at 2.7-3.2x the floor.

Time shares used for ranking (of 143 s):
| bucket | seconds | share |
|---|---|---|
| decode body GEMMs (Marlin) | 2,130 x 18.6 = 39.6 | 28% |
| decode all-reduce + gathers | 2,130 x 9.8 = 20.9 | 15% |
| decode lm_head GEMMs (target + 2 draft) | 2,130 x 4.3 = 9.2 | 6.4% |
| decode GDN state | 7.8 | 5.4% |
| decode rest (attn, MTP layers, small kernels, idle) | ~6.6 | 4.6% |
| mixed: prompt GEMMs (Marlin) | 104.5K x 0.19 = 19.9 | 14% |
| mixed: prompt all-reduce | 104.5K x 0.10 = 10.5 | 7.3% |
| mixed: decode part | 14.8 | 10% |
| mixed: eager/CPU overhead E | ~13 | 9% |
