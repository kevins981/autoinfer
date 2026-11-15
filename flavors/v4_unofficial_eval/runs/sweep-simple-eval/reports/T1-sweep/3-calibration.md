# T1 sweep, step 3: napkin cost model, calibrated against the champion (BASE)

Measured source: `task/baseline/nsys_c16/profile.sqlite` (BASE, 64 ShareGPT prompts, c=16, node-level
graph trace), per-step medians on GPU 0 with `scripts/step_anatomy.py` and my per-kernel variant
`agents/leader-1/step_kernels.py` (same step definition: NVTX execute_context ranges). Marlin
bandwidth: `task/baseline/ncu_marlin/marlin_c16_key.csv`. Whole-benchmark: `task/baseline/official`.
No new GPU job was needed for calibration.

Napkin constants: BW 1.0 TB/s, F 165 TFLOP/s (bf16 / FP8-fp32acc), Link 25 GB/s per direction,
per-collective latency ~10 us.

## Decode step (16 requests x 3 rows = 48 rows; measured period 38.5 ms, GPU busy 37.0 ms)
| part | napkin | measured | ratio | note |
|---|---|---|---|---|
| FP8 linear (Marlin), 264 calls | 12.4 GB / 1.0 TB/s = 12.4 ms | **18.6 ms** | 1.50 | ncu: 0.63-0.81 TB/s per call (small out_proj 16 MB worst); memory-bound (48 rows << ridge 82) |
| All-reduce, 135 calls x 0.49 MB | 135 x (0.49 MB/25 GB/s + 10 us) = 4.0 ms | **7.75 ms** (57 us each) | 1.9 | NCCL RING_LL via host |
| GDN recurrent (48 layers) | state r+w 2 x 1.57 MB x 16 x 48 = 2.4 GB -> 2.4 ms | 2.95 ms (+0.19 conv) | 1.2 | near floor |
| target lm_head, 48 rows | 1.27 GB -> 1.27 ms | 1.59 ms | 1.25 | bf16 cuBLAS |
| logits all-gather, target 48 rows | 11.9 MB/25 GB/s = 0.48 ms | 0.95 ms | 2.0 | gathers full V/2 x 48 rows |
| draft lm_head (1 call seen per step, 16 rows) | 1.27 ms | 1.37 ms | 1.08 | cutlass wmma |
| draft logits all-gather | 4 MB -> 0.16 ms | 0.33 + 0.06 ms | 2.4 | |
| full attention (16 layers) + KV write | KV ~16 x 500 tok x 2 KB/GPU x 16 layers = 0.26 GB -> 0.26 ms | 0.42 ms | 1.6 | |
| norms, act, rope, gates, small bf16 GEMMs (in_proj_ba) | ~0.2 GB -> 0.3 ms | ~1.3 ms | ~4 | launch-size-bound tiny kernels |
| sampling / bookkeeping / mamba align | ~0 | ~0.3 ms | | |
| GPU idle in step | 0 | 1.3 ms (nsys, node trace inflates) | | |
| **total** | **~23.6 ms** | **38.5 ms** | 1.63 | |

Unprofiled check: official median ITL 39.4 ms (= one decode step) -> the nsys step time is
representative (decode steps are graph-replayed, little nsys distortion).

## Mixed prefill step (median 215 prompt + 45 decode rows = 260 rows; measured period 146.8 ms)
| part | napkin | measured | ratio | note |
|---|---|---|---|---|
| FP8 linear (Marlin), 524 calls | 2 x 12.4 G x 260 = 6.4 TFLOP / 165 = 39 ms | **59.2 ms** | 1.52 | compute-bound (260 > ridge 82), 65% of bf16 peak |
| All-reduce, 135 calls x 2.66 MB | 135 x (2.66 MB/25 GB/s + 10 us) = 15.7 ms | **62.5 ms** (463 us each = 5.7 GB/s) | **4.0** | the largest napkin miss |
| GDN prefill (chunked) + decode | ~3 ms | 7.4 ms | 2.5 | |
| lm_heads (target + draft rows) | ~2.6 ms | 4.3 ms | 1.7 | |
| logits all-gathers | ~0.6 ms | 1.7 ms | 2.8 | |
| elementwise/index/norm/act (eager: no CUDA graph above 96 tokens, no torch.compile) | ~1 ms | ~5-6 ms | ~5 | |
| GPU idle | 0 | ~6 ms (eager launches; nsys inflates) | | |
| **total** | **~62 ms** | **146.8 ms** | 2.4 | |

## Whole benchmark (official BASE: 635.9 tok/s, ~91K output tokens => ~143 s)
In the profile, 64 prompts produced 50 mixed steps and 501 decode steps (50 x 146.8 + 501 x 38.5 =
26.6 s, the capture length). Scaling to 400 prompts: ~310 mixed steps x 146.8 ms = **~46 s (32%)**
and ~2,520 decode steps x 38.5 ms = **~97 s (68%)**; total ~143 s, consistent with the official
time. Tokens per decode step 16 x 2.345 (official acceptance) = 37.5.

## What the calibration says
1. The napkin's ranking is right for decode memory (weights dominate) but it **underestimates
   communication by 2x (decode) to 4x (prefill)**: NCCL over host memory reaches only ~5.7 GB/s on
   2.66 MB messages and ~8.6 GB/s on 0.49 MB. This is an implementation-layer efficiency, but it is
   the single largest gap between design and measured time (~47 ms per mixed step, ~3.7 ms per
   decode step). In the framework's terms, effective Link_BW here is a function of message size
   and transport, not a constant.
2. Marlin reaches ~67% of BW in steady decode (vs ~80% in isolation under ncu): a 1.5x miss on the
   biggest single cost; a kernel at 90% BW would save ~4.8 ms per decode step.
3. The vocab dimension (lm_head + logits all-gathers) is ~4.3 ms per decode step (11%), which the
   framework's standard template does not show (07 §4.3 lists it as an open question).
4. Mixed steps are 4x longer than decode steps although they occur once per request: 32% of time.
   Their cost is half communication.
5. Small eager kernels (no compile, no graphs above 96 tokens) cost ~5-6 ms per mixed step and
   ~1.3 ms per decode step.
With these efficiency factors applied (comm x2-4, Marlin x1.5, small ops measured), the model
reproduces both step types within ~5%; rankings below use the measured part times.
