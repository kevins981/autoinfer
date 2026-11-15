# T2 sweep, step 3: napkin cost model, calibrated against the champion (BASE)

## Measurements used
- **M1** `agents/leader-2/prof_graph/` (GPU job 2, T2): BASE, 100 ShareGPT prompts at c=16,
  `TRACE=graph` (exact launch/idle timing). Bench under profiling: 599 tok/s (lib.sh's unprofiled
  100-prompt BASE reference: ~596) -> the profile is representative. `steps.txt`/`steps.csv`
  (task/scripts/step_anatomy.py, GPU 0).
- **M2** `task/baseline/nsys_c16/` (node-level trace, 64 prompts): kernel-level contents of one
  decode step and of mixed steps (scripts `agents/leader-2/{onestep,seq,layer,prefill,gaps}.py`).
- **M3** `task/baseline/ncu_marlin/marlin_c16_key.csv`: isolated Marlin times/bytes.

## Where the time goes (M1, 37.9 s captured)
| Step kind | steps | median period | total | share |
|---|---|---|---|---|
| decode (M ≈ 45-48 rows) | 644 | 38.59 ms | 23.6 s | **62%** |
| mixed (prefill + decode rows) | 85 | 138.0 ms | 14.3 s | **38%** |

Mixed steps are only 12% of steps but 38% of time. The napkin must model both.

## Model of a decode step (M = 48 rows, per GPU)
Napkin rule: memory-bound terms = bytes / BW with BW = 1.0 TB/s; link terms = 20 µs latency +
bytes / 12 GB/s (the observed NCCL ring-over-host rate); compute is below the ridge (N_ridge≈82 > 48).

| Part | Napkin (formula) | Napkin | Measured (M2, one 48-row step, GPU 0) | Ratio |
|---|---|---|---|---|
| Body FP8 weights (Marlin) | 12.16 GB / BW | 12.2 ms | 18.6 ms (195 x 49 µs + 69 x 132 µs) | 0.65 |
| GDN state | 48 l x 16 r x 1.57 MB x (1 read + **3 writes**, one per verified row) = 4.8 GB / BW | 4.8 ms | 3.66 ms kernel (76 µs x 48; 100 MB in 76 µs > DRAM peak -> writes land in L2 and drain during the next kernels) | — |
| Marlin + GDN together | (12.16 + 4.8) GB / BW | 17.0 ms | 22.3 ms | **0.76** |
| All-reduce (128 x 0.49 MB) | 128 x (20 µs + 0.49 MB / 12 GB/s) | 7.8 ms | 8.2 ms (132 x 61 µs) | 0.95 |
| lm_head x3 (target 48 rows, 2 draft steps) | 3 x 1.27 GB / BW | 3.8 ms | 4.3 ms (1.59 + 1.37 + 1.37) | 0.88 |
| Logit all-gathers (target 48 rows x 124k x 2 B = 11.9 MB; drafts 16 rows = 4 MB each) | 20 µs + bytes / 12 GB/s | 1.0 + 0.35 + 0.35 | 0.96 + 0.33 + 0.33 | ~1.0 |
| Attention (KV fp8, ~6.4k tokens x 16 KB) + KV write | 0.1 GB / BW | 0.1 ms | 0.5 ms | 0.2 (launch-bound small kernels) |
| norms, act, conv, small bf16 GEMMs, bookkeeping | (small bytes, ~400 launches x ~2 µs) | ~0.8 ms | ~1.3 ms | |
| GPU idle (CPU gaps between draft steps) | 0 | 0 | 0.99 ms (M1) | |
| **Step** | | **36.0 ms** (with Marlin+GDN at 0.76) | **38.6 ms** (M1 median) | **0.93** |

Per-layer in situ (M2), GDN layer: qkvz 57 µs, GDN 76, out_proj 32, AR 62, gate_up 141, down 62, AR 62
(+ norms/conv/act ~15) ≈ 510 µs. Isolated (M3, ncu): qkvz 53.6, out 25.9, gate_up 115, down 60
-> in situ Marlin is 5-25% slower than isolated; the largest gap (gate_up +26 µs) follows an AR,
out_proj (+6 µs) follows the GDN kernel's state writes.

Interpretation: the decode step is calibrated to ~7%. The one large "efficiency factor" is
Marlin+GDN at 0.76 of DRAM peak. Everything with N=1 reuse (state, lm_head) runs near peak.

## Model of a mixed step
Napkin: T_mixed = max(T_GPU, T_CPU) where
- T_GPU = Marlin(M) + AR(M) + other, Marlin(M) ≈ compute at ~80% of bf16 peak once M > ~100
  (24.3 GFLOP/row/GPU; 165 TFLOP/s), AR(M) = 128 x (20 µs + M x 10 KB / 12 GB/s), other ≈ 10 ms
  (GDN chunked prefill, attention, norms) + ~6 ms (target head, drafts).
- T_CPU = eager launch of the whole model: **mixed steps above 96 tokens are not CUDA-graphed**
  (`max_prefill_cudagraph_capture_tokens: 96`, `max_cudagraph_capture_size: 96` in the server
  log; compilation mode NONE), M2 shows every kernel launched individually, ~2 ms per layer.

| Bucket (prefill tokens) | n | napkin T_GPU | napkin T_CPU | measured period | measured Marlin | measured NCCL | measured GPU idle |
|---|---|---|---|---|---|---|---|
| < 100 (M≈140) | 28 | Marlin 21 + AR 17.5 + 16 = ~55 ms | ~125 ms | 133.5 | 32.8 | 29.1 (incl. waits) | 52.7 |
| 100-200 | 13 | ~70 ms | ~125 ms | 135.2 | 41.5 | 26.4 | 48.3 |
| 200-300 (M≈300) | 6 | Marlin 55 + AR 35 + 16 = ~106 ms | ~125 ms | 136.8 | 60.6 | 35.6 | 23.2 |
| 300-500 | 18 | ~135 ms | ~125 ms | 151.5 | 76.4 | 45.2 | 7.0 |
| > 500 | 20 | ~230 ms | ~125 ms | 244.7 | 131.4 | 80.7 | 6.4 |

Marlin in mixed steps: 6.07 s for ~33.1k rows -> 33.1k x 24.3 GFLOP / 165 TFLOP/s = 4.9 s at
100% of bf16 peak: Marlin runs at **~80% of compute peak** there (not weight-re-read bound).

Interpretation: small and medium mixed steps are **CPU-launch-bound** (the "idle" column, plus
rank skew hidden inside NCCL waits: the AR kernel of the early rank spins until the late rank
arrives, so NCCL's 29 ms at M≈140 is ~2x the napkin 17.5 ms). Large mixed steps are GPU-bound,
with the all-reduce ~1/3 of the step (12 GB/s over the host link).

## What the framework's cost model missed (structural)
1. **Host launch rate is a resource.** (F, BW, Link_BW) cannot explain 60-70 ms of the 133 ms
   small mixed steps; a 4th resource "CPU dispatch" (or the implementation fact "graph-replayed or
   not") is needed. Here it is worth ~8% of total time.
2. **Deferred writes:** a kernel's HBM writes can drain during later kernels (GDN 3-way state
   writes -> slower out_proj). Per-window accounting must charge bytes written to the window where
   they drain, or Marlin looks worse and GDN looks better than they are.
3. **Link efficiency** on this box is ~12 GB/s algbw (half the PCIe spec) + ~20 µs per exchange.

## Calibration summary (napkin vs measured)
- Decode step: napkin 36.0 ms vs 38.6 ms measured (0.93), parts within 5-15% except attention/small kernels.
- Mixed step: napkin max(T_GPU, T_CPU) within ~10% for all buckets (e.g. 125 vs 133.5; 106-125 vs 136.8; 230 vs 244.7).
- Total time (M1): decode 644 x 36.0 + mixed sum(napkin) ≈ 23.2 + 13.2 = 36.4 s vs 37.9 s measured.
