# T1 sweep, step 3: napkin cost model, calibrated against the champion (BASE)

Measured run: task/baseline/nsys_c16/profile.sqlite (BASE, 64 ShareGPT prompts, c=16, node-level graph
trace). Step split by the task's `scripts/step_anatomy.py` (501 decode steps, 50 mixed steps, GPU 0);
per-kernel numbers from my script `agents/leader-1/decode_step.py` on the median step of each kind.
Napkin inputs: BW 1.0 TB/s, link 25 GB/s per direction via host, bf16 tensor 165 TFLOP/s.

## 3.1 Where the benchmark time goes
| | count (64 prompts) | median period (nsys) | share of capture |
|---|---|---|---|
| decode steps (M = 42 rows: 14 req x 3) | 501 | 38.5 ms | 19.3 s = **72.5%** |
| mixed steps (one ~260-330 token prompt + ~45 decode rows) | 50 | 147-149 ms | 7.4 s = **27.5%** |
The capture produced 64x228/27 s ≈ 540 tok/s vs 636 official (nsys overhead ~15%, mostly on eager
mixed steps). Client median ITL (official) 39.4 ms ≈ nsys decode period, so decode steps are barely
distorted (CUDA graphs).
Throughput identity: tok/s ≈ (16 x acceptance 2.345) / step_time, diluted by mixed steps:
16 x 2.345 / 38.5 ms = 975 tok/s if decode-only; x0.725 time share ≈ 707 → minus the drop of steps with
<16 running → ~636 measured. Consistent.

## 3.2 Decode step (M = 42), per GPU: napkin vs measured
| part | napkin (bytes or work → time) | measured (median step) | efficiency / note |
|---|---|---|---|
| body FP8 GEMMs (Marlin), 64 layers x 4 | 12.2 GB / 1 TB/s = **12.2 ms** | **18.6 ms** (264 launches) | 0.66 of spec BW in-situ. ncu of the same kernels (isolated): 0.77-0.82 (e.g. gate_up 91 MB in 115 us); in-situ the same GEMM takes 141 us. Unexplained 20% in-situ gap (clocks? L2 state after NCCL?) — itself a finding |
| all-reduce, 128 body + ~7 drafter | 135 x (430 KB / 25 GB/s = 17 us + ~5-10 us sync) = **3.0-3.6 ms** | **7.7 ms** (57-60 us each) | NCCL ring LL via host SHM achieves ~7.4 GB/s on 430 KB; 2.1-2.5x the napkin |
| logits all-gather, target (42 rows) | 10.4 MB / 25 GB/s = **0.42 ms** | **0.84 ms** | 12 GB/s |
| logits all-gather, 2 draft steps (14 rows) | 2 x 3.5 MB / 25 GB/s = **0.28 ms** | **0.59 ms** (2 x 0.29) | |
| target lm_head (bf16, 1.27 GB) | **1.27 ms** | **1.46 ms** | 0.87 |
| draft lm_head x2 (same 1.27 GB each) | **2.54 ms** | **2.73 ms** | 0.93 |
| GDN recurrent update, 48 layers | state fp32 1.57 MB/req/layer, 14 req, read+write: 48 x 44 MB = 2.1 GB → **2.1 ms** | **2.96 ms** (+0.19 conv1d) | 0.71 |
| full attention, 16 layers (+ KV write, merge) | ~0.1 GB KV → **0.1 ms** | **0.59 ms** | latency-bound tiny kernels (~30 us each) |
| MTP layer x2 (fc + attn + MLP, ~0.2 GB) | **0.4 ms** | **~0.9 ms** incl. its ~4 ARs/AGs | |
| norms, act, conv, small bf16 GEMMs (in_proj_ba), sampling bookkeeping | ~350 launches x ~2-3 us = **~0.8 ms** | **~1.5 ms** | |
| GPU idle inside the step | 0 | **1.3 ms** (step_anatomy) | CPU work between graphs (draft metadata rebuilt eagerly, sampling) |
| **total** | **~23.6 ms** (spec rates) | **38.5 ms** | |

Calibrated model (napkin x measured efficiency): Marlin 0.66 BW; NCCL all-reduce 58 us at 430 KB
(≈ 15 us + size / 8.4 GB/s); all-gather ≈ size / 12 GB/s; bf16 lm_head 0.9 BW; GDN 0.71 BW; per-launch
floor ~3 us; idle 1.3 ms. With these factors the model reproduces 38.5 ms within 1 ms, part by part.
Largest gaps napkin → measured: Marlin (+6.4 ms), all-reduce (+4.1 ms), small kernels + idle (+2 ms),
logit gathers (+0.7 ms), GDN (+0.9 ms).

## 3.3 Mixed step (≈375 rows: 330 prompt + 45 decode), per GPU
| part | napkin | measured | note |
|---|---|---|---|
| body GEMMs (Marlin, compute-bound now) | 375 x 2 x 12.2 G = 9.2 TFLOP / 165 TF = **55 ms** | **81 ms** (519 launches: Marlin splits M>256 into 2 calls) | 113 TFLOP/s = 0.68 of bf16 peak. FP8 W8A8 (330 TF peak) napkin: **28 ms** |
| all-reduce x133, 3.84 MB each | 3.84 MB / 25 GB/s = 154 us → **20.5 ms** | **39.1 ms** (294 us each, 13 GB/s) | 1.9x napkin |
| GDN chunked prefill + decode update | ~4 ms | **~9.6 ms** (fused_sigmoid_gating 4.4 ms for the 45 decode rows!, chunk kernels ~3.5 ms) | the decode rows in a mixed step use a slower non-MTP kernel (83 us vs 62 us/layer) |
| lm_head + drafter + gathers | ~4.5 ms | **~5 ms** | |
| other (copies, gathers, eager launches) | ~1 ms | **~6 ms** + 6.4 ms idle | mixed steps run eager (no full graph): launch overhead visible |
| **total** | **~85 ms** (bf16) | **147-149 ms** | |

## 3.4 What the calibration says about the cost model
- The framework's `max(F, BW, Link)` per window is right in shape (decode memory-bound, mixed
  compute-bound) but **every window here is a strict sequence**: the 128 combine nodes sit on the
  serial chain and nothing overlaps them today, so `Total = Σ` of parts, not max. The link term is
  2x its napkin because NCCL's LL protocol over host SHM is latency/protocol-bound, not bandwidth-bound.
- The framework's model has no term for **implementation efficiency of a node** (Marlin 0.66,
  NCCL 0.4-0.5 of link, launch floor). These dominate the gap to the speed of light (38.5 vs ~24 ms).
  I carry them as measured factors.
- Ranking consequence: the parts with the largest *absolute* excess over napkin are
  (1) Marlin efficiency, (2) all-reduce protocol, (3) lm_head bytes repeated 3x per step (a
  missing-index sharing failure, see 2-graph), (4) mixed-step compute at bf16 rates.
