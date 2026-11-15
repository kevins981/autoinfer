# T3 sweep, step 3: napkin cost model, calibrated

Measured: nsys profile of the champion of the time, T1-014 (GPU job 72; `agents/leader-3/jobs/prof_champ/`,
64 ShareGPT prompts at c=16, node-level graph trace), plus microbenchmarks (jobs 73, 74, 75 below).
Scripts: `agents/leader-3/scripts/{budget,exposed,one_step,gemm_shapes}.py`.
T2-008 (the champion since 21:23) = T1-014 + GDN DeltaLog; its effect is noted where it matters.

Hardware constants used by the napkin: HBM 1.0 TB/s; PCIe 25 GB/s per direction (measured 25-26
GB/s one direction, copy engine, job 75); FP8 dense fp32-acc 330 TFLOP/s.

## A. Decode step (16 requests x 3 rows = 48 rows, no prefill), per GPU
198 steps, `budget.py`: median GPU span **33.6 ms** (step period 31.3 ms median over all 493
decode steps of any width in `steps.txt`).

| Part | Napkin (bytes / rate) | Napkin ms | Measured ms (calls x avg) | Measured / napkin |
|---|---|---|---|---|
| gate_up GEMM (89.1 MB, W8A8 Triton, grid 408) | 64 x 89 MB / 1 TB/s | 5.70 | 8.53 (65 x 131 us) | 1.50 (0.68 TB/s) |
| down (44.6 MB) + out_proj/o_proj (15.7 MB) (grid 240) | 64 x 60.3 MB | 3.86 | 5.15 (130 x 39.6) | 1.33 |
| qkvz (41.9 MB, grid 384) | 48 x 41.9 MB | 2.01 | 2.57 (48 x 53.4) | 1.28 |
| full-attn qkv (36.7 MB, grid 336) | 16 x 36.7 MB | 0.59 | 0.80 (17 x 46.9) | 1.36 |
| **body GEMMs total** | 12.2 GB | **12.2** | **17.05** | **1.40** (72% of peak BW) |
| target lm_head (bf16 1.27 GB, cuBLAS) | 1.27 GB | 1.27 | 1.58 | 1.25 |
| draft lm_head (fp8 0.64 GB, CUTLASS) x 2 draft steps | 1.27 GB | 1.27 | 1.53 | 1.21 |
| MTP layer GEMMs (M=16) | ~0.2 GB | 0.2 | ~0.25 | ~1.2 |
| **all-reduce** (hostmem kernel), 135 calls x 480 KB | 2 x 480 KB / 25 GB/s = 38 us each | 5.2 | **6.56** (135 x 48.6 us) | 1.26 |
| GDN MTP kernel: fp32 state 1.57 MB/req/layer, 1 read + 3 writes | 48 x 16 x 6.3 MB | 4.8 | 3.48 (48 x 72.5) | 0.72 (L2 absorbs part of the writes) |
| in_proj_ba (bf16, splitK) on a side stream | 0.5 MB | ~0.1 | 1.16 (overlapped, not on the critical path) | - |
| small kernels (quant 264x2.4, norm 135x3.1, conv, act, attention 18x14.6, rope, ...) | ~0 bytes; launch floor ~2 us each | ~1.2 | ~2.1 | ~1.7 |
| gaps between kernels | 0 | 0 | 0.73 (`one_step.py`, step 100) | - |
| **Sum** | | **~26.9** | **~33.3** (span 33.6) | **1.25** |

Notes:
- The step's kernel times sum to 34.0 ms, of which ~1.2 ms (ba GEMM side stream) overlaps; the
  sum reproduces the measured span within 1%.
- After T2-008's DeltaLog the GDN row drops by ~1.2 ms/step (T2's ABBA +3.9%); the T2-008 decode
  step is then ~32.3 ms span.
- Microbenchmark (job 73, real per-GPU shapes, GDN kernel alone, 48 layer instances): 120 us per
  call with 3 state writes vs 59 us with only row 0 written. In the server the kernel is faster
  (72.5 us) because the writes drain later (and slow the following GEMM: T2's finding).

### All-reduce, calibrated separately (job 74, `ar_bench.py`, 2 ranks, isolated)
| rows | bytes | NCCL us | hostmem S=1 (champion) us | hostmem pipelined S=2/4/8 |
|---|---|---|---|---|
| 16 | 0.16 MB | 37 | 27 | 29 / 46 / 63 |
| 48 | 0.49 MB | 147 | **53** | 55 / 75 / 93 |
| 128 | 1.31 MB | 132 | 128 | 116 / 145 / 146 |
| 256 | 2.6 MB | 242 | 254 | 218 / 269 / 247 |
| 1024 | 10.5 MB | 770 | 831 | 829 / 869 / 853 |
| 2048 | 21 MB | 1473 | 1638 | 1631 / 1646 / 1641 |

- Observed: AR time = ~14 us + bytes / ~13 GB/s, for both NCCL and hostmem at large sizes.
  Sub-chunk pipelining of write-to-host and read-from-peer (S) doesn't help.
- PCIe (job 75, copy engine, pinned): one direction 25-26 GB/s on each GPU; D2H and H2D at once on
  one GPU 22.6 GB/s total. The AR moves 480 KB out and 480 KB in per GPU per call: 960 KB / 22.6
  GB/s = 42 us + latency, so the 48.6 us decode AR is **at the link's practical floor**; the
  napkin's 25 GB/s per direction overstates the link (full duplex isn't available here).
  Corrected napkin: AR(n rows) ~ 10 us + 2 x n x 10 KB / 22.6 GB/s.

### GEMM efficiency (job 75, isolated Triton W8A8, M=48)
Back-to-back repeated GEMM: gate_up 107 us, down 51, qkvz 50, qkv 37, out_proj 35 us.
In the server: gate_up 131, (down+out_proj avg) 39.6, qkvz 53.4, qkv 46.9.
Observed: gate_up reaches only 0.68 TB/s in the server (0.83 isolated hot); down reaches ~0.88.
(L2-prefetch variants of this test were confounded by my L2 flush, which leaves dirty lines;
no conclusion drawn from them.)

## B. Steps that contain a prefill chunk ("mixed")
49 such steps in the profile, median 231 prefill tokens + 45 decode rows, median period 82 ms
(decode-only: 31 ms). `exposed.py`: per step, AR time not overlapped by any other kernel:

| prefill tokens | period ms | non-AR busy ms | AR exposed ms | AR kernel sum ms |
|---|---|---|---|---|
| 57-129 | 71-78 | 41-51 | 15-20 | 15-20 |
| 231-358 | 82-92 | 43-53 | 9-11 | 15-35 |
| 482-885 | 222-276 | 67-120 | 49-130 | 98-167 |
| 1846 | 352 | 211 | 92 | 181 |

Napkin for a mixed step of n rows (per GPU): compute n x 24.4 GFLOP at ~250 TFLOP/s (Triton FP8)
plus weights 12.2 ms if larger; AR = 128 x (10 us + n x 20 KB / 22.6 GB/s).
- n = 276 (231+45): compute ~27 ms (weights 12 ms are below that), AR (if not hidden) ~36 ms.
  Measured non-AR busy 52.5 ms (~1.9x napkin), AR exposed 10.7 ms.
- n = 527 (482+45): compute ~51 ms; AR ~57 ms if not hidden at all. Measured: non-AR 67 ms but
  **AR kernels sum 167 ms** (256 NCCL calls of ~2.7 MB, i.e. ~650 us each = ~4 GB/s). In
  isolation NCCL does 2.6 MB in 242 us. So in the server, the NCCL ARs above the 1.31 MB hostmem
  cap run ~2.7x slower than isolated, and most of it is exposed.
- Interpretation: the mixed steps' excess is mainly (1) large NCCL all-reduces that run far below
  the link rate in the server (CPU proxy / SM contention: not yet known which), and (2) GEMMs at
  about half the napkin's compute rate. Messages <= 1.31 MB (<= 128 rows) take the hostmem path;
  the row-split halves of a >= 263-row step exceed it and go to NCCL.

## C. Whole benchmark (official trial 1 of T1-014: 108.0 s, 90.9K output tokens)
- Steps: ~2726 (iteration_tokens histogram), ~234 with > 128 tokens.
- If decode-only steps cost ~31.5 ms (profile), ~2490 of them take ~78 s, leaving **~30 s
  (~28%) for the ~234 mixed steps (~126 ms each, ~450 prefill tokens on average)**.
- Napkin for those mixed steps (450+45 rows): compute ~48 ms, AR ~50 ms hideable behind it:
  ~50-55 ms ideal vs ~126 ms measured. Decode: ~24.5 ms napkin (at measured link floor and
  DeltaLog GDN) vs ~32 ms measured.

## Where the napkin is wrong for this machine (framework notes)
1. The framework's window cost treats `Link_BW` as one rate. Here the link isn't full duplex
   (22.6 GB/s total for D2H+H2D on one GPU vs 25-26 one way), and an all-reduce between two
   processes without P2P goes through host memory (write, then the peer reads): its cost is
   2 x bytes over one shared link, plus ~10-14 us latency per call. 135 calls per decode step
   make the latency term (~1.5 ms) a resource of its own.
2. "Shared reads counted once per window" is optimistic for the 72 MB L2: the GDN state writes
   (T2's finding) and the dirty-line drain show that bytes written can be charged to the next
   window, not the one that wrote them.
3. GEMM efficiency at M=48 is 0.68-0.88 of HBM peak, shape-dependent (gate_up is worst). The
   framework's bandwidth-bound rows need an efficiency factor per shape to rank kernel work.
4. Under nsys, eager (non-graph) mixed steps show ~18 ms of GPU idle per step; PROFILING.md warns
   this is inflated. Not used as a finding without a non-nsys check.
