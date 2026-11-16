# T3 sweep, step 3: napkin cost model, calibrated against the champion

Measurement: GPU job 44, nsys of the champion (workspace t3-champ = T1-007, its serve_args), 64 ShareGPT prompts at c=16,
node-level graph trace. Files: `agents/leader-3/prof1/` (profile.sqlite, breakdown.txt, steps.txt/csv). Per-kernel per-step
numbers from `agents/leader-3/stepk.py` (same step definition as `task/scripts/step_anatomy.py`: an `execute_context` NVTX
range to the next one, GPU 0). nsys slows eager work (prefill steps) somewhat; decode steps are CUDA-graph replays.

Napkin constants: DRAM 1.0 TB/s, FP8 TC 165 TFLOP/s (fp32 accumulate), bf16 165 TFLOP/s, PCIe via host ~25 GB/s/dir.

## A. Decode step (prefill 0, 44-48 rows; 250 steps): period 36.0 ms (median 35.97), kernel sum 35.15 ms

| part | napkin (bytes or flops / rate) | measured ms/step | ratio meas/napkin | note |
|---|---|---|---|---|
| FP8 body GEMMs, target (64 layers) | 12.17 GB / 1 TB/s = 12.2 | 16.9 | 1.39 | gate_up g816: 130 us for 89.1 MB = 0.69 TB/s; down+out/o g240: 0.77 TB/s; qkvz g384 0.79; attn qkv g336 0.78 |
| target lm_head (bf16, cuBLAS) | 1.27 GB = 1.27 | 1.58 | 1.25 | 0.80 TB/s |
| drafter lm_head, 2 steps (bf16, cutlass wmma 8x970 grid) | 2 x 1.27 = 2.54 | 2.71 | 1.07 | 1356 us each |
| drafter body (MTP layer x2, fc bf16) | 2 x (0.2 + 0.05) = 0.5 | ~0.5 | ~1 | inside g816/g240/g336 counts (+1 each) and Kernel2 |
| GDN in_proj_ba (bf16 [48x5120]x[5120x48], split-K 27 + splitKreduce) | 0.5 MB -> ~1 us x 48 | 1.19 (6.8 + 18.1 us per layer) | **~25x** | **corrected (review T3-1, worker job 45): off the critical path** — runs on side stream s17 concurrently with the qkvz GEMM (overlaps it in 11,999/12,000 instances, zero tail); isolated it costs 5.8 us |
| all-reduce, HMAR, 135/step (128 target + MTP + misc) | 0.49 MB each way at 25 GB/s = 20 us; x135 = 2.7 | 6.22 (46.0 us each) | 2.3 | latency-bound: write-to-host + flag + read-from-host serial (~2 x 20 us) |
| GDN decode recurrence (48 layers) | read 1 + write 3 states (all 3 window slots) x 16 req x 1.57 MB = 100 MB/layer -> 4.8 ms | 3.38 (70.4 us each) | **0.70** | faster than the napkin's DRAM floor: the napkin's write traffic is wrong somewhere (fewer slots written, or write-back absorbed by L2). Open: ncu dram bytes (cheap test) |
| per-token-group FP8 quant (264/step) | 48x5120 bf16 read: ~0.5 us each | 0.64 (2.4 us) | ~5 | launch-floor kernels |
| fused add+RMSNorm (132), act_and_mul (66), conv1d (48), qk-norm/rope/gate | ~1 us each | 0.83 | — | launch-floor kernels |
| attention (FlashInfer, 16 + 2 MTP calls, fp8 KV) | ~16 KB/token x ~500 tok x 16 = 0.13 GB -> 0.13 | 0.39 | 3 | small |
| ncclSendRecv once per step (454 us) + 2-3 small all-gathers | ~0 | 0.53 | — | batch-sharded sampling / bookkeeping exchange; likely absorbs rank skew (a wait, not work) |
| other (resample, logits stats, mamba pre/post copy, fills) | — | 0.25 | — | |
| GPU idle in the step window | 0 | **2.05** (period minus union of kernel intervals; corrected, review T3-1) | — | kernel sum 35.15 ms double-counts ~1.18 ms of overlap (mostly ba/qkvz) |
| **total** | **floor ~17.5 (all at roofline, AR serial)** | **36.0** | 2.06 | |

Napkin with measured efficiencies (GEMMs at 0.77 TB/s, AR at 46 us, others as measured) reproduces 35.2 ms of kernels + 0.8 idle:
the model is calibrated part by part, except GDN (napkin over-estimates by 1.4x: flagged).

Where the 18.5 ms gap to the floor sits: GEMM efficiency 4.7 ms, all-reduce latency 3.5 ms (and 2.7 ms of AR even at the link
floor, all on the critical path), drafter lm_head bf16 2.7 ms (an fp8 or smaller head would be ~1.3 ms), ba-proj overhead 1.2 ms,
small-kernel floors ~1.5 ms, SendRecv/idle ~1.3 ms.

## B. Mixed prefill steps (prefill > 0; 52 steps): median 260 rows (215 prefill + 45 decode), period median 120.7 ms, mean 142.7 ms

| part | napkin | measured ms/step (mean) | note |
|---|---|---|---|
| NCCL all-reduce (RING_LL), 129.5/step | 260 x 5120 x 2 B = 2.66 MB; via host at 25 GB/s/dir: ~110 us pipelined, ~210 us store-and-forward; x129 = 14-27 ms | **47.1 (363.5 us each)** | 7.3 GB/s effective: NCCL's LL protocol over SHM. The largest single gap in the run |
| FP8 GEMMs (Triton W8A8, compute-bound at M>=~100) | at mean 422 rows: 10.3 TFLOP / 165 = 62 ms | **50.1** (all W8A8 kernels; corrected) | ~205 TFLOP/s achieved: 165 is not the ceiling (T2 measured up to 230) |
| GDN (decode rows via fused_sigmoid_gating 47 x 81 us; chunked prefill kernels) | — | ~5 | |
| target lm_head + bf16 small GEMMs | — | ~4.8 | |
| quant/norm/act/elementwise | — | ~4.6 | eager-ish (more launches) |
| GPU idle | 0 | 9.5 (median) | CPU launch/scheduling gaps on mixed steps |

## C. Whole benchmark
- From the profile's step mix: 52 prefill steps per 64 prompts (some steps carry several chunks). For 400 prompts: ~325 mixed steps
  x 121-143 ms = 39-46 s, plus ~91.2K output tokens / (16 x 2.34 per step) = ~2,440 decode steps x 36 ms = 88 s: **127-134 s
  napkin vs 118 s measured** (91.2K / 771.3 tok/s). Over by 8-13%, consistent with nsys inflating the eager parts of mixed steps.
- Share of wall time (from the napkin): **decode ~68%, mixed prefill steps ~32%**. Inside mixed steps, NCCL all-reduce is ~35-39%.
  So: mixed-step all-reduce ~12% of the benchmark, decode all-reduce ~12%, decode FP8 GEMMs ~32%, drafter lm_head ~5.5%,
  GDN decode ~6.5%, ba-proj overhead ~2.3%.
- Tokens per step: acceptance 2.34 (champion eval) -> the decode cost per output token is 36 / (16 x 2.34) = 0.96 ms.

## D. Calibration findings (mismatches as findings)
1. Mixed-step all-reduce runs at 7.3 GB/s vs ~25 GB/s possible: NCCL LL over SHM is ~3x off the link. Biggest napkin gap after GEMM efficiency.
2. GDN decode kernel is faster than its napkin DRAM floor if all 3 slots are written: the napkin's write model is wrong; check before building a
   "write fewer states" (DeltaLog) candidate on it.
3. The bf16 `in_proj_ba` GEMM costs ~25x its napkin (split-K + reduce kernel launched at M=48, N=48).
4. HMAR at 46 us for 0.49 MB is ~2.3x the one-crossing link time: the two PCIe crossings are serialized.
5. FP8 GEMMs at decode reach 0.69-0.79 TB/s (gate_up the worst).

## E. Corrections after review T3-1 (adopted)
- Mixed-step AR: rank-0 mean 363.5 us, rank-1 412.7 us, mean start offset 93.5 us across 6,732 paired collectives; at 244/276 actual rows
  rank-0 averages 217.6/248.2 us, close to the bare ~205 us at 260 rows. My "~160 us of skew" subtracted a 260-row bare number from a
  422-row mean: wrong. Skew exists but is not priced.
- Whole-benchmark napkin: I counted all output tokens in pure-decode steps and added mixed steps that also decode (~10.8K tokens, ~10 s):
  that double count, not nsys, explains most of the 8-13% overprediction.
- GDN: the kernel does write all 3 slots (fused_gdn_decode_kernel.cu:323-334; slots are real per-request blocks, worker's code reading of
  gdn_attn.py:465/486). Cold microbench (worker, job 45/46): 120/89/60 us for 3/2/1 writes at ~845 GB/s, vs 70 us in the server: the
  difference is L2 write-back attributed to later kernels / warm state, not fewer writes.
- Mixed-step idle: median 9.5 ms but mean 22.7 ms.
