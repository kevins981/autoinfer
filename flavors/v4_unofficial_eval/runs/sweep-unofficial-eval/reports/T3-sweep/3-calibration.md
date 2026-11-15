# T3 sweep, step 3: napkin cost model, calibrated against HEAD (T1-016)

Measurements (T3 jobs, workspace t3-head = T1-016, HEAD's serve_args, 64 ShareGPT prompts at c=16):
- job 153: nsys, CUDA graphs traced per node -> `agents/leader-3/prof_head_node/` (kernel names inside graphs)
- job 155: nsys, graph-level trace -> `agents/leader-3/prof_head_graph/` (accurate periods / idle)
- per-kernel tables: `agents/leader-3/decode_kernels_grid.py` / `k2.py` (my scripts; step definition = PROFILING.md's
  NVTX `execute_context_*` ranges, as in step_anatomy.py)
- HEAD's unofficial run (bank/T1-016/unofficial/1/b): 808.9 tok/s, median ITL 29.0 ms, acceptance 2.33.

## 0. Where the benchmark's time goes (job 155, graph trace, GPU 0)
| step kind | steps | sum of periods | share | median period |
|---|---|---|---|---|
| decode-only (16 req x 3 rows) | 484 | 12.86 s | **63%** | 26.7 ms (29.3 at exactly 16 req; real-bench median ITL 29.0) |
| mixed (a prefill rides a decode step) | 52 | 7.49 s | **37%** | 112.6 ms (95 ms for ~60-token prefills, 115 ms at ~350, 160-220 ms at 600-900) |
Extra cost of a mixed step over a decode step: ~70-190 ms, on average **~117 ms per admitted prompt**.
Cross-check against the real bench (no nsys): 200 prompts, 56.9 s; ~1230 full decode steps x 29 ms = 35.7 s, leaving
~21 s = ~105 ms per prompt for mixed steps and tail. Consistent: **prefill admission costs ~1/3 of wall time**,
although prefill FLOPs at the compute floor would be ~15% (SPEED_OF_LIGHT: ~16 s of ~108 s).

## 1. Decode step (M = 48 rows), per GPU: napkin vs measured (job 153, 195 steps with exactly 16x3 rows)
Napkin: bytes / 1.008 TB/s (spec) for memory-bound parts; link 25 GB/s per direction.
| part | napkin basis | napkin | measured | ratio | note |
|---|---|---|---|---|---|
| Marlin GEMMs (MLP, GDN in_proj, attn qkv, MTP x2) | 11.49 GB weights | 11.4 ms | **15.10 ms** (198 calls, 76 us avg) | 1.32 | 76% of spec BW |
| host all-reduce (135 calls, 0.49 MB each) | 0.49 MB/25 GB/s = 20 us if write||read; 40 us if serial | 2.7 ms (overlapped) / 5.3 (serial) | **5.74 ms** (42.5 us avg) | 2.1 / 1.08 | matches the serial write-then-read model: kernel has 4 blocks, write -> flag -> read |
| GDN recurrence (fused decode, replay mode) | S read 1.21 GB + write 0.6-1.2 GB | 1.8-2.4 ms | **1.91 ms** (48 x 39.8 us) | ~1 | at its byte floor for the chosen keep set |
| target lm_head (bf16, 1.27 GB/GPU) | 1.27 GB | 1.26 ms | **1.59 ms** | 1.26 | cuBLAS ampere_bf16 128x64 |
| Triton W8A16 out_proj / o_proj (64+1 calls) | 1.04 GB | 1.03 ms | **1.40 ms** (21.2 us avg) | 1.36 | |
| draft lm_head fp8 (98,304 rows), 2 steps | 2 x 0.25 GB | 0.50 ms | **0.63 ms** | 1.25 | cutlass e4m3 GEMM |
| NCCL SendRecv (1/step) + AllGather (5/step) | bytes ~0 | ~0.05 ms | **0.57 ms** | — | sync points: absorb inter-GPU skew + launch latency |
| fused add+RMSNorm (132), act_and_mul (66), conv1d (48) | ~0.1 GB | 0.1 ms | **0.75 ms** | — | launch-latency floor ~3 us each |
| attention (FlashInfer, 18 calls) + KV write + merge | KV ~0.1-0.2 GB | 0.15 ms | **0.48 ms** | 3 | small, latency-bound |
| GDN ba proj (bf16 cutlass 48 + splitK 48) | 48 x 0.5 MB | 0.03 ms | **0.31 ms** | — | 2 launches of ~3 us per layer |
| MTP misc GEMMs, sampling, bookkeeping | | 0.1 ms | ~0.35 ms | | |
| **sum of kernels** | | **~19.6 ms** (AR overlapped) | **28.8 ms** | 1.47 | GPU idle in step ~0.15 ms (graph trace): the decode step is GPU-bound |
Calibrated factors: streaming GEMMs at M=48 reach 74-80% of spec DRAM BW; every small kernel costs >= 3 us;
the host AR costs (write + read) serial = 2 x bytes/25 GB/s + ~3 us.
With these factors the napkin reproduces the step: 11.49/0.76 + 1.04/0.74 + 1.27/0.8 + 0.5/0.8 + 1.9 + 135 x 42.5 us
+ ~250 small kernels x 3 us + 0.6 (sync) = 15.1 + 1.4 + 1.6 + 0.6 + 1.9 + 5.7 + 0.75 + 0.6 = **27.7 ms vs 28.8 measured**.

## 2. Mixed step, small prefill (~60 prompt tokens + 15 decode req = ~105 rows), job 153/155, 16-19 steps
| part | napkin | measured (GPU 0) | note |
|---|---|---|---|
| Marlin (M ~105) | weights once: 11.4/0.76 = 15 ms (compute 2.4 TFLOP = 15 ms) | **27.7 ms, 381 calls** | **Marlin processes <= 64 rows per pass: M=105 -> 2 passes -> weights read twice** (198 -> 381 calls) |
| host AR (M ~105, 1.07 MB) | 2 x 43 us serial / 43 overlapped | **146.6 us avg x 127 = 18.6 ms** | 7.3 GB/s: far below the link; 4 blocks can't keep PCIe busy at this size |
| W8A16 Triton (M ~105) | 1.4 ms | 3.0 ms (50 us avg) | untuned mid-M configs |
| GDN mixed path (chunked prefill kernels + spec-decode update + replay flush) | ~2 ms | ~6 ms | non-fused path |
| CPU: worker execute_model wall (NVTX) | | **~116 ms** | ~1,700 eager launches per step (~36 per GDN layer: ~24 torch index/gather/elementwise + ~12 Triton), 68 piecewise graph launches |
| GPU busy / period | | 65 ms / 95 ms (graph trace) | **CPU-bound under nsys** (nsys inflates CPU time; without nsys probably partly CPU-bound) |

## 3. Mixed step, typical prefill (~300-400 prompt tokens + 15 decode req, M ~350-450)
| part | napkin | measured | note |
|---|---|---|---|
| Marlin | compute 2 x 11.5e9 x 353 = 8.1 TFLOP / 165 TF = 49 ms; weights re-read ceil(353/64)=6x = 69 GB -> 68 ms | **57 ms (median), 67 ms at 355-400** | between the compute floor and the 6x re-read floor (L2 catches some re-reads) |
| all-reduce: NCCL ring (rows > 128) | 3.6 MB: ring sends 3.6 MB per rank -> 144 us at 25 GB/s | **329 us avg x 134 = 44 ms** | 11 GB/s; the chunked GEMM+AR overlap of HEAD is eager-only and these steps (<= 512 tokens) are graphed, so it does not apply |
| GDN chunked prefill etc. | | 7.4 ms | |
| other / idle | | ~10 ms / ~0 | GPU-bound at this size |
| **period** | ~55 ms (compute floor + hidden AR) | **~115 ms** | |

## 4. Calibration verdict
- Decode: the napkin with measured efficiency factors reproduces 27.7 vs 28.8 ms. Remaining gap to the pure
  byte floor (~16 ms + AR at link speed 2.7 ms = ~19 ms) is: GEMM bandwidth efficiency (~4.3 ms), AR serialization
  (~3 ms), many small kernels and sync points (~2 ms).
- Mixed steps: the napkin "prefill is compute-bound" (SPEED_OF_LIGHT) is wrong for this engine at these sizes:
  the cost is dominated by **communication (NCCL / 4-block host AR at low bandwidth), Marlin's 64-row passes
  re-reading weights, and CPU launch overhead in the eager GDN path**, not by FLOPs. A mixed step costs 3.5-7x a
  decode step for 0.2-1x a decode step's worth of extra rows.
