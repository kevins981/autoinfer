# T1 sweep, step 3: napkin cost model, calibrated against BASE

Measured: task/baseline/nsys_c16/profile.sqlite (64 prompts, c=16), per-kernel per-step means from
`agents/leader-1/step_kernels.py` (an adaptation of task/scripts/step_anatomy.py: same step
definition, GPU 0, kernels grouped by demangled name + grid) and step_anatomy.py medians;
ncu key metrics task/baseline/ncu_marlin/marlin_c16_key.csv; official trial1 bench.log.

## Decode step (M ~42 rows; nsys median period 38.5 ms; real median ITL 39.3 ms)
| part | napkin | measured (nsys, ms/step) | note |
|---|---|---|---|
| body FP8 GEMMs (264 Marlin launches) | 12.5 GB / 1.0 TB/s = 12.5 ms (floor); at Marlin's ncu-measured ~0.8 TB/s: 15.6 | 17.5 | ncu: gate_up 91 MB in 115 us (0.79 TB/s), qkvz 43 MB in 54 us (0.80), down 46 MB in 60 us (0.77), out 16 MB in 26 us (0.63). Compute at M=42: 42 x 27 GFLOP = 1.1 TFLOP / 165 = 6.9 ms < memory: memory-bound |
| lm_head x ~3 (target + 2 MTP steps) | 3 x 1.27 GB / 0.93 TB/s = 4.1 | 2.96 (cutlass wmma 2.2/step) + 0.78 (ampere 0.5/step) + 0.21 (gemv 0.2/step) = 3.95 | ~10% of the step; 2/3 of it is draft-side |
| GDN delta rule (48 layers) | state r+w: 16 x 48 x 1.57 MB x 2 = 2.4 GB -> 2.4 ms | 2.2 (gdn_decode_post_conv_mtp, all grids) + conv 0.1 | at the memory roofline |
| all-reduce (134.8/step) | 430 KB per AR; PCIe 25 GB/s ring(2): S bytes each way -> 17 us + ~10 us latency = ~27 us -> 3.6 ms | **6.2 (46 us each)** | NCCL RING_LL, 2 channels, via host (SHM): ~1.7x napkin |
| all-gather (4.9/step) | logits 42 x 124,160 x 2 B = 10 MB target, ~4 MB per draft step -> ~0.7 ms at 25 GB/s | 1.1 | |
| full attention + KV write (16 layers + MTP) | KV 16 req x ~400 tok x 2 KB x 16 = 0.2 GB -> 0.2 ms | 0.4-0.85 | small |
| norms, act, rope, small GEMMs, misc | — | ~1.0 | |
| GPU idle in step (launch gaps, CPU) | — | 1.3 | |
| **total** | **~33 (napkin with measured efficiencies) ; 25.5 at pure roofline incl. AR** | **38.5** | |

## Mixed prefill step (~215 prefill tokens + ~45 decode rows; nsys median period 146.8 ms)
| part | napkin | measured |
|---|---|---|
| body GEMMs (Marlin, 524 launches) | 260 rows x 27 GFLOP/GPU = 7.0 TFLOP / 165 TFLOP/s = 42 ms (compute-bound: 260 > ridge 82) | **91.9** (46% of bf16 peak) |
| all-reduce (135) | 2.66 MB each; ~106 us at 25 GB/s -> 14 ms | **64.9 (481 us each = 5.5 GB/s)** |
| GDN chunked prefill + misc | — | ~8 |
| lm_head + AG | — | ~6 |
| GPU idle | — | 6.4 |
| **total** | **~75** | **146.8** |

## Whole benchmark (one trial: 143.2 s, 91,080 output tokens)
- Prefill steps: the profile has 50 mixed steps for 64 new prompts (0.78/prompt) -> ~312 for 400
  prompts x 147 ms = **~46 s (32%)**.
- Decode steps: the rest of the tokens: each step makes ~16 x 2.345 = 37.5 tokens; mixed steps also
  emit ~35 decode tokens -> decode steps ~ (91,080 - 312 x 35) / 37.5 ~ 2,140 x 38.5 ms = **~82 s (57%)**.
- Remaining ~15 s (10%): ramp-down tail (fewer than 16 requests at the end, ~228-token outputs with
  long-tail lengths), scheduling gaps, nsys-vs-real differences. Napkin total 128 s vs 143 s
  measured: **reproduces ~90%**, good enough to rank.

## Where the time goes (napkin shares of the 143 s)
| bucket | seconds | share |
|---|---|---|
| decode body GEMMs | 2,140 x 17.5 ms = 37 s | 26% |
| prefill-step GEMMs | 312 x 92 ms = 29 s | 20% |
| **all-reduce + all-gather, decode** | 2,140 x 7.3 ms = 16 s | 11% |
| **all-reduce, prefill steps** | 312 x 65 ms = 20 s | 14% |
| lm_heads, decode (2/3 draft) | 2,140 x 4.0 ms = 8.5 s | 6% |
| GDN + attention | ~7 s | 5% |
| misc, idle, tail | ~26 s | 18% |

## Calibration verdicts (what the napkin got wrong, itself a finding)
1. **All-reduce is 1.7x (decode) to 4.5x (prefill) slower than the PCIe napkin.** 5.5 GB/s for
   2.66 MB messages means NCCL's LL protocol over SHM with 2 channels is far from the link's
   ~25 GB/s. This gap (~25 s of 143 s at stake) is the biggest napkin-vs-measured mismatch.
2. **Prefill GEMMs run at 46% of bf16 peak** (Marlin is a decode kernel; dequant + bf16 MMA).
3. Decode GEMMs at ~80% of DRAM BW: the gap to roofline is ~5 ms/step (13% of a decode step).
4. lm_head is read 3x per step at bf16; the framework's "W_lm missing k" is not exploited because
   draft steps are serial.
