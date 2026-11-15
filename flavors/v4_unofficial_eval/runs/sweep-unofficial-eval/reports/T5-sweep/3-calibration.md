# T5 sweep, step 3: napkin cost model, calibrated against HEAD (T3-004)

Measurements (all mine, workspace t5-head = T3-004, HEAD's serve_args):
- **job 201**: nsys, 64 ShareGPT prompts, c=16: node trace (`agents/leader-5/prof/head_node/`) and graph trace (`head_graph/`).
- **job 202**: nsys **NVTX-only** (no CUDA tracing, near-zero overhead), 200 prompts, c=16 (`head_nvtx/`): 843.7 tok/s (HEAD's unofficial starts: 863 / 874), so it is representative.
- Step = one `execute_model` NVTX range (PROFILING.md definition); period = start to next start on the same TP worker. Scripts: task/scripts/step_anatomy.py, agents/leader-5/scripts/decode_kernels.py, inline sqlite queries.

## 1. Where the benchmark's time goes (job 202, 200 prompts, 54.1 s)
| step kind | steps | time | share | median period |
|---|---|---|---|---|
| pure decode (16 req x 3 rows = 48 tokens) | 767 | 21.2 s | 39% | 27.6 ms |
| other pure decode (tail, < 16 req) | 458 | 10.6 s | 20% | 20.8-27.7 ms |
| **mixed (prefill + decode)** | **183** | **22.4 s** | **41%** | **101.6 ms** |

Mixed-step cost vs prefill size (extra period over a 27.6 ms decode step):
| prefill tokens | steps | median period | extra |
|---|---|---|---|
| 1-99 | 69 | 96.8 ms | 69 ms |
| 100-199 | 30 | 100.3 | 73 |
| 200-399 | 31 | ~101-103 | 72-75 |
| 400-599 | 25 | 165 | 138 |
| 600-999 | 23 | 186 | 158 |

**Observation:** the extra is ~70 ms and flat from 1 to 400 prefill tokens; it is not proportional to prefill work. CPU time inside execute_model (NVTX range length): decode 2.5 ms (graph replay, async), mixed 96-98 ms ≈ the whole period. In the CUDA trace (job 201), a mixed step issues ~1,800 launches (decode: 22), runs as ~64 piecewise CUDA-graph pieces (0.56 ms each) with the GDN/attention prefill path eager in between, and starts with a **blocking D2H copy** (FlashInfer metadata's `seq_lens.cpu()` in mixed batches: `flashinfer.py` ~l.1730), so the CPU cannot prepare it while the previous step runs.

## 2. Pure decode step (M = 48), napkin vs measured (job 201, node trace, 207 steps, medians per step, GPU 0)
| part | napkin input | napkin (1.0 TB/s, 25 GB/s PCIe) | measured | ratio |
|---|---|---|---|---|
| body weight GEMMs (W8A16 Triton) + MTP layer x2 | 12.6 GB | 12.6 ms | **15.9 ms** (gate_up 113 us = 789 GB/s; down/out 815; qkvz 753; qkv 737) | 1.26 |
| target lm_head (cuBLAS bf16) | 1.27 GB | 1.27 | 1.58 | 1.24 |
| draft heads (fp8, 98K rows) x2 | 0.50 GB | 0.50 | 0.63 | 1.26 |
| GDN fused decode (state read + 1/2 write, lazy) | 1.8 GB | 1.8 | 1.83 | 1.0 |
| full attention + KV write | ~0.15 GB | 0.15 | 0.35 | ~2 |
| **all-reduce (host-pinned, 135/step, 491 KB each)** | 66 MB/direction over PCIe | 2.6 ms if write/read pipelined, 5.3 if not | **5.70** (42 us each) | 2.2 (1.07 vs non-pipelined) |
| **in_proj_ba (cuBLAS Kernel2 + splitKreduce, 48 layers)** | 24 MB bf16 | 0.03 | **0.88** | ~30 |
| norms, act, conv, rope, gather, misc (~350 small kernels) | ~0.1 GB | ~0.3 | ~1.3 | ~4 |
| **sum** | | **19.6 ms** | **28.2 ms** sum of kernel medians; period 25.0 (graph trace), 27.6 (job 202) | |

Calibration factors: weight streaming ~0.79 of spec BW (1.26x napkin). The GDN kernel is at spec BW (state reads are perfectly coalesced). The host all-reduce matches the **non-pipelined** napkin (write all, flag, then read all; grid 4; `csrc/host_all_reduce.cu`). in_proj_ba is a pure-overhead outlier.

## 3. Mixed step, napkin vs measured (M ≈ 48 + P rows)
| part | napkin (P = 165) | measured |
|---|---|---|
| weights / compute (W8A8 at M ≥ 80; 24.3 GFLOP/token/GPU) | max(12.2 GB / 0.8 TB/s = 15 ms, 213 x 24.3 GF / 200 TF = 26 ms) ≈ 26 ms | W8A8 kernels ~22 ms at M=106 (≈ 560 GB/s effective: below W8A16's 790) |
| all-reduce: 213 rows x 10 KB x 128 | 280 MB/dir: 11 ms pipelined host / 28 ms NCCL LL (≈10 GB/s measured: 325 us per 3.2 MB) | rows ≤ 122 host-AR (1.25 MB cap), else NCCL |
| GDN prefill (chunked delta rule) + conv + attention | ~3-5 ms GPU | ~5-8 ms GPU |
| lm_head + drafter | ~3 ms | ~3 ms |
| **sum (GPU-bound napkin)** | **~45 ms (P=165), ~35 ms (P=60)** | **~100 ms** (CPU-bound: ~1,800 launches + blocking sync) |

=> the napkin over-predicts nothing here; it under-predicts mixed steps by ~55-65 ms each. That gap is **CPU launch/python time**, a resource the framework's (F, BW, Link_BW) vector does not have (see 4-knobs.md §0).

## 4. Whole benchmark: napkin vs measured (200 prompts)
| | napkin | measured |
|---|---|---|
| decode steps 1,225 | 1,225 x ~20-26 ms ≈ 27-32 s | 31.8 s |
| mixed steps 183 | 183 x ~40 ms ≈ 7.3 s | 22.4 s |
| total | ~35-39 s → 1,170-1,300 tok/s | 54.1 s → 844 tok/s |

The two largest napkin-vs-measured gaps, in seconds of the 54 s run: **mixed-step CPU overhead ~11-13 s (~22%)**, then **decode all-reduce ~2.3 s pipelining + ~2.5 s overlap headroom**, **decode GEMM efficiency ~2.6 s**, in_proj_ba ~0.7 s, small kernels ~0.8 s.
