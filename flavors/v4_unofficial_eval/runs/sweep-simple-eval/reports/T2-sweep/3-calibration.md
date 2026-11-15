# T2 sweep, step 3: napkin cost model, calibrated against BASE

Measured data: task/baseline/nsys_c16 (BASE, 64 prompts, c=16, per-kernel trace), split into steps
with task/scripts/step_anatomy.py (`--csv` -> agents/leader-2/base_steps.csv) and my per-kernel step
dump (agents/leader-2/step_dump.py, step_seq.py; one full decode step's kernel list:
agents/leader-2/decode_step_seq.txt). ncu Marlin numbers: task/baseline/ncu_marlin/marlin_c16_key.csv.
Napkin constants: BW 1.0 TB/s, bf16-MMA 165 TFLOP/s, FP8-MMA 330 TFLOP/s, PCIe ~25 GB/s/direction.

## A. Whole benchmark (official: 400 prompts, 91.2K output tokens, 635.9 tok/s = 143.4 s)
Step model from the profile (GPU 0, step_anatomy definition):
- decode step with 16 requests (48 rows): **39.9 ms** median period.
- prefill-containing step: linear fit over 50 steps, **period = 88.1 ms + 0.244 ms x (rows in step)**.
Reconstruction: decode-equivalent steps = 91.2K / (16 x 2.345) = 2,430 x 39.9 ms = 97.0 s;
prefill steps' fixed excess over a decode step (88.1 - 39.9 = 48.2 ms) x ~400 = 19.3 s;
prompt rows 104K x 0.244 ms = 25.4 s. **Total 141.7 s vs 143.4 s measured (-1%).** Good enough to rank.

| share of the benchmark | s | % |
|---|---|---|
| decode steps (steady, 48 rows) | 97.0 | 68 |
| prefill steps: fixed excess over a decode step | 19.3 | 13 |
| prefill steps: per prompt row | 25.4 | 18 |

## B. Decode step (48 rows), napkin vs measured, per GPU
| part | napkin | measured (ms/step) | ratio | note |
|---|---|---|---|---|
| target body GEMMs (Marlin, 12.2 GB fp8) | 12.25 | 18.1 | 1.48 | ncu in isolation 77-80% of 1008 GB/s (43 MB in 53.6 us, 91 MB in 115 us); in-situ ~68% (small GEMMs 16 MB at 63%) |
| MTP layer GEMMs (2 draft steps) | 0.4 | 0.5 | 1.25 | |
| target lm_head (1.27 GB bf16) | 1.27 | 1.59 | 1.25 | cuBLAS ampere_bf16 128x64 |
| draft lm_heads (2 x 1.27 GB) | 2.54 | 2.74 | 1.08 | CUTLASS wmma 16x16 tile, 16 rows |
| all-reduce, 135 x 491 KB (bf16 48x5120) | 3.4 (25 us each at 25 GB/s incl. latency) | 8.16 (median 58-64 us) | 2.4 | ring LL; p5 = 14-19 us, so latency is not the floor; includes cross-rank wait |
| logits all-gather (target 11.9 MB, drafts 2 x 4 MB) | 0.8 | 1.67 | 2.1 | target AG 0.95 ms |
| GDN recurrence (48 layers; fp32 S: read 25 MB + write 3 x 25 MB per layer) | 4.8 | 3.66 (+0.19 conv) | 0.76 | **napkin over-predicts**: 100 MB in 76 us would be 1.3 TB/s. Either fewer slot writes happen than I assume or L2 absorbs part: open, ncu needed |
| full attention + KV write (16 layers, ~0.1 GB KV) | 0.2 | 0.85 | 4 | launch/occupancy bound small kernels (18 x ~20 us) |
| norms, act, rope, small bf16 GEMMs, sampling bookkeeping | 0.3 | 1.3 | 4 | 300+ tiny kernels |
| GPU idle inside the step (host gaps) | 0 | 1.3 | - | mostly 2 gaps of 120-240 us at step start (input prep) |
| **total** | **26.0** | **39.9** | **1.53** | |

Interpretation: the decode step is bound by weight bytes (body + 3 lm_head reads = 16.5 GB napkin,
22.4 ms measured) plus a communication term that is 2.4x its bandwidth napkin. The two biggest
ratio gaps in ms: Marlin efficiency (5.9 ms) and all-reduce (4.8 ms).

## C. Prefill-containing step (~260 rows), napkin vs measured
| part | napkin | measured (fit: fixed + per row) | note |
|---|---|---|---|
| GEMMs | 24.4 GFLOP/row / 165 TF = 0.148 ms/row (bf16 MMA); 0.074 with W8A8 FP8 | 13.9 ms + 0.178 ms/row | Marlin's marginal rate = 137 TFLOP/s (83% of bf16 peak): Marlin is fine at large M; FP8 MMA would halve the floor |
| all-reduce | 135 x 10 KB/row / 25 GB/s = 0.054 ms/row | **40.3 ms** + 0.060 ms/row | per-row part matches bandwidth; the **fixed 40 ms** (vs 8-10 ms in decode) is cross-rank waiting |
| GPU idle | ~0 | **20.3 ms** | step runs eager (mode NONE, graphs only <= 96 tokens): host-launch bound |
| period | ~39.9 (decode part) + 0.2/row | 88.1 + 0.244/row | **fixed excess ~48 ms per prefill step = 13% of the benchmark** |

Interpretation: a mixed step above 96 tokens runs eager: ~1,500 kernel launches from Python on each
rank, the ranks drift apart, and each of the 135 all-reduces waits for the slower rank. Neither the
GPU's F nor BW explains these 48 ms: **the framework's resource vector (F, BW, Link_BW, C) has no
CPU-launch / host-step resource**, which is where 13% of this benchmark goes. I add it as a 4th
resource `Host` (launches per step x ~5-10 us when not graph-captured, + the cross-rank skew it
induces on every combine node).
Caveat: nsys inflates eager launch cost (PROFILING.md); but the benchmark reconstruction (A) uses the
same profile and lands within 1% of the official time, so the inflation is small here or offset.

## D. Speculation numbers (official results.json)
Acceptance length 2.345 = 1 + 0.78 + 0.565 (per-position acceptance 0.783 / 0.568). Conditional
acceptance of draft 2 given draft 1: 0.568/0.783 = 0.73.
