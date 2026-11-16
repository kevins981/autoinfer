# T6 sweep, step 3: napkin cost model, calibrated on the champion T3-007

Measurement: nsys node-level trace of T3-007 (job 98, 64 prompts at c=16, `agents/leader-6/prof1/`: steps.txt from step_anatomy.py,
per-kernel split per step kind by my query of profile.sqlite, GPU 0). Node tracing inflates times somewhat; use ratios.
Hardware constants: DRAM 1.0 TB/s, FP8 TC 165 TFLOP/s (fp32 acc) -> ridge ~165 FLOP/B -> **N_ridge ~ 82 rows for FP8 weights** (not 330:
the framework's 4090 example uses the fp16-accumulate rate, which the accumulation rule forbids for W8A8), PCIe via host ~25 GB/s per direction.

## Decode step (506 steps; median 39 decode rows = ~13 requests x 3; period 27.5 ms profiled; ITL p50 31 ms unprofiled incl. interleaved prefills)
| part | napkin (per GPU) | measured kernel ms/step | note |
|---|---|---|---|
| body FP8 GEMMs (W8A8 Triton) | 12.2 GB / 1 TB/s = 12.2 ms (N=39 < 82: BW-bound) | 16.15 | 76% of DRAM peak; L2 prefetch already helps |
| L2 prefetch side stream | (overlaps AR) | 4.54 (side stream) | not on the critical path |
| all-reduce (HMAR, ~123/step) | 123 x (~12 us latency + 0.4 MB/25 GB/s) ~ 3.4-4.5 ms | 5.40 (44 us each) | +2 per MTP step included |
| other GEMM: target lm_head bf16, FP8 draft lm_head x2, MTP layer x2, GDN ba (bf16) | 1.27 + 1.28 + 0.4 + ~0.1 = ~3.0 ms | 4.45 (cutlass Kernel2 3.59 + ampere bf16 0.85) | |
| GDN recurrent kernel (MTP, 3 rows/request) | state traffic 13 req x 24 heads x 64 KB x (1 read + **3 writes**) x 48 layers = 3.8 GB -> 3.8 ms | 2.38 | kernel time < napkin: part of the write-back is absorbed by L2 and paid later by the next GEMMs (write-back), so the true cost is likely between 2.4 and 3.8 ms |
| attention (Triton unified), KV write | ~0.1-0.3 GB | 0.33 | |
| norms/quant/act/conv/etc | ~1 ms | ~1.2 | |
| **sum on critical stream** | **~25 ms** | **~30.8 kernel (minus prefetch); period 27.5, idle 0.36** | napkin within ~10% |

## Prefill/mixed step (50 steps; median 215 prefill + 45 decode rows = M ~260; period 119 ms, GPU busy 90, idle 28)
| part | napkin | measured ms/step | note |
|---|---|---|---|
| body FP8 GEMMs | 260 rows x 25.6 GFLOP = 6.7 TFLOP / 165 = 40 ms (compute-bound, N > 82) | 54.7 | 73% of FP8 peak |
| all-reduce (NCCL RING_LL, ~136/step at ~2.6 MB) | 136 x 2.66 MB / 25 GB/s = 14.5 ms | **57.4 (421 us each)** | **4x the napkin: the largest napkin-vs-measured gap in the system** |
| GDN chunked prefill + spec part (Triton) | ~5 ms | ~7 | |
| other | ~5 ms | ~10 | |
| GPU idle inside the step | 0 | 28 ms | node trace inflates eager launches; needs a graph-level trace to judge |
| **total** | **~65 ms** | **119 ms period** | |

## Whole benchmark (400 prompts)
Scaling the profile's step mix (50 prefill : 506 decode for 64 prompts + 16 warmups) to 400 prompts: ~310 prefill steps and ~3,160 decode steps.
Napkin with the measured per-step times unprofiled (decode ~25 ms, prefill ~100 ms): 3,160 x 25 + 310 x 100 = 79 + 31 = ~110 s vs measured 106.6 s
(bank/T3-007/eval/1 trial1). **Prefill/mixed steps are ~28% of wall time and their NCCL all-reduces ~12% of wall time.**

## Where the napkin and the measurement disagree (findings)
1. Prefill all-reduce: 4x over the link napkin (RING_LL protocol through host memory without P2P; LL sends 8-byte data+flag pairs, halving bandwidth, and the SHM transport adds a copy).
2. Decode GEMMs at 76% of DRAM peak (the remaining 24% ~ 3.9 ms/step is a kernel-efficiency gap).
3. GDN state: kernel time below the byte napkin (L2 absorbs writes); its writes still cost DRAM bandwidth later.
4. Prefill-step GPU idle 28 ms: unexplained at this trace level.
