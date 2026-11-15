# T1 sweep, step 3: napkin cost model, calibrated against BASE

Measured data: `task/baseline/nsys_c16/profile.sqlite` (BASE, 64 ShareGPT prompts, c=16, kernel-level trace),
split into steps with `task/scripts/step_anatomy.py` (`--csv`, my copy: agents/leader-1/steps.csv) and
per-kernel with my scripts `agents/leader-1/step_kernels.py`, `step_seq.py` (GPU 0). ncu: `baseline/ncu_marlin/marlin_c16_key.csv`.
Official run: `baseline/official/trial1/bench_c16.json` (duration 143.2 s, 91,080 output tokens, 104,526 input tokens).

Hardware rates used by the napkin: BW 1.008 TB/s, F 165 TFLOP/s (bf16, fp32 acc), link 25 GB/s per direction (spec; worker t1-w1 is measuring it).

## A. Decode step, 16 requests x 3 rows (M = 48), GPU 0 — mean of the 186 steps with exactly 48 decode tokens
| part | napkin (bytes or FLOPs -> time) | measured | measured / napkin | notes |
|---|---|---|---|---|
| body GEMMs (Marlin, 264 calls) | 12.2 GB + MTP 2x0.2 GB = 12.6 GB -> 12.5 ms | 18.6 ms | 1.49 (BW eff 0.67) | big GEMMs reach ~0.80 TB/s (ncu: 43/16/91/46 MB in 54/26/115/60 us); small ones and attention qkv lower |
| all-reduce (135 calls, 491 KB each, 16-row ones in the drafter) | 135 x 491 KB / 25 GB/s -> 2.7 ms | 8.2 ms (60 us/call) | 3.0 | NCCL RING_LL over host SHM |
| all-gather of logits (target 48 rows: 6 MB/GPU; 2 draft x 16 rows: 2 MB/GPU; + 2 small) | 10 MB / 25 GB/s -> 0.4 ms | 1.7 ms (960 + 2x330 us + small) | 4.2 | consumer only needs argmax |
| lm_head (target M=48 + 2 draft M=16; same 1.27 GB/GPU weight read 3x) | 3.8 GB -> 3.8 ms | 4.3 ms (1.60 + 2x1.37) | 1.14 | already near BW |
| GDN recurrent update (48 layers, fp32 state read+write 50 MB/layer) | 2.4 GB -> 2.4 ms | 3.7 ms (+0.2 conv) | 1.55 | 76-89 us/layer vs 50 us floor |
| full attention (16 layers, KV ~7 MB total) | ~0.01 ms | 0.5-0.9 ms | latency-bound | 17-18 calls of ~20 us |
| norms/act/rope/gates, small bf16 GEMMs, bookkeeping | ~0 (bytes tiny) | ~1.5 ms | launch/latency-bound | ~400 small kernels |
| GPU idle inside the step | 0 | 1.4 ms | - | CPU prep between graphs; inflated by nsys node tracing |
| **step total** | **~22 ms raw napkin** | **40.0 ms period (38.6 busy)** | | |

Calibrated decode model (per-class efficiency fitted above): T_dec(M) ~ 12.6 GB/(0.67 BW) + 135 x AR(M) + AG + 3.8 GB/(0.88 BW) + GDN(M) + ~3 ms fixed latency-bound work.
AR(M) for NCCL LL here ~ 60 us at M=48 (bytes/8 GB/s + ~0); the raw napkin underestimates the link term 3x: **this is the largest napkin/measured mismatch in the step and itself a finding** (either the link is slower than 25 GB/s in practice, or NCCL LL wastes it; t1-w1's microbenchmark will tell which).

## B. Prefill-containing step (mean of 50: 395 prompt rows + 42 decode rows = ~437 rows)
| part | napkin | measured | ratio |
|---|---|---|---|
| body GEMMs: 2 x 437 x 12.6e9 FLOP = 11 TFLOP (compute-bound: 437 >> ridge 82) | 66.7 ms | 91.9 ms | 1.38 (73% of 165 TF) |
| all-reduce: 128 x 437 x 10 KB = 573 MB | 22.9 ms | 66.6 ms (~520 us/call, ~4.3 GB/s) | 2.9 |
| GDN chunked prefill + decode | ~2.4 ms (state) + small FLOPs | 8.4 ms | 3.5 |
| lm_head + other | ~2 ms | 8.5 ms | |
| attention | <1 ms | 2.6 ms | |
| GPU idle | 0 | 15.0 ms | eager (not graphed) launches, inflated by nsys |
| **total** | **~95 ms** | **194.7 ms period** | 2.0 |

## C. Whole benchmark (official BASE: 143.2 s)
Model: N_pf prefill-containing steps ~0.78 per prompt (profile: 50 for 64 prompts) -> ~312; they produce ~14 x 2.345 tokens each;
decode steps produce the rest: (91,080 - 312 x 33)/(16 x 2.345) ~ 2,150 steps.
- decode: 2,150 x 38.5 ms (nsys busy, ~graph-replayed so little nsys distortion) = 82.8 s
- prefill-containing: 312 x 194.7 ms = 60.7 s
- **predicted 143.5 s vs measured 143.2 s** (the profile's own 64-prompt bench: steps sum 27.17 s vs bench duration 27.26 s).
So the calibrated model reproduces the run. Shares of the official time:
| | seconds | share |
|---|---|---|
| decode steps | ~83 | 58% |
| prefill-containing steps | ~61 | 42% |
| of which communication (AR+AG), both kinds | 2,150 x 9.8 ms + 312 x 66.6 ms = 21.1 + 20.8 = **41.9** | **29%** |
| of which Marlin GEMMs | 2,150 x 18.6 + 312 x 91.9 = 40.0 + 28.7 = 68.7 | 48% |
| GDN | 8.4 + 2.6 = 11 | 8% |
| lm_head + logit gathers (decode) | 2,150 x 6.0 = 12.9 | 9% |

Caveat: prefill-step idle (15 ms) is partly nsys overhead on eager launches (PROFILING.md); the period-based model still matches the official duration, so the true idle is probably smaller and the other parts slightly larger than shown, or the official run's prefill steps are a bit faster than the profile's.
