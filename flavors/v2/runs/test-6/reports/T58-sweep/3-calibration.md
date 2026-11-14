# T58 sweep, step 3: napkin cost model, calibrated against the champion

Phase 1. Measurements: worker leader-58.w1:
- **Job 485:** nsys of T52-005 at c=16, 96 prompts, graph and node traces. Files: `agents/leader-58.w1/prof/SUMMARY.md`.
- **Jobs 490/491:** microbenchmarks. Files: `agents/leader-58.w1/calib/SUMMARY.md`, `B_tables.md`.
- **Step log:** 400 prompts, `calib/steplog.md` (§4).

## 1. Calibrated hardware constants (measured, per GPU; GPU 0 and GPU 1 agree within ~1%)

| Constant | Spec / my prior (1-inputs.md) | Measured | Consequence |
|---|---|---|---|
| DRAM read BW | 1008 GB/s | **957 GB/s** (copy kernel) | weights 12.18 GB -> 12.7 ms floor |
| FP8 mma e4m3, f32 accumulate | 165 TF (SPEED_OF_LIGHT) | **329 TF** | my prior was 2x low. Ridge for W8A8 = 329/(2 x 0.957) = **~172 rows** |
| FP8 mma e4m3, f16 accumulate | 330 | **657 TF** (2.0x f32-acc) | only usable if the accumulation rule allows it (§5 of 4-knobs) |
| BF16 mma, f32 accumulate | 82.6 | **165 TF** | Marlin bf16-activation ridge ~86 rows |
| bs_gemm (champion W8A8 prefill GEMM), cold | — | 255-265 TF at M=512, 283-290 at M=2048 (86-88% of f32-acc peak) | prefill GEMM is near its ceiling in isolation |
| Decode-size GEMMs (M <= 128), weights cold in L2 | — | **580-670 GB/s = 60-70% of DRAM** (bs_gemm and t31) | decode GEMMs are latency/occupancy-limited, not DRAM-limited, when cold |
| Same, weights hot in L2 | — | qkvz 46-52 us (vs 70 cold), gate_up 104-108 (vs 138), down 54-57 (vs 79) | L2 helps ~1.3-1.5x, not 3x: the kernels are not L2-BW-limited either |
| GPU 0 vs GPU 1 speed | GPU 0 locked 2235 MHz, "~15% slower on compute" (ledger) | ratio 0.99-1.01 per GEMM shape in situ; 1.013 median in microbench | **no measurable rank skew**; the asymmetric-split idea (05 §4.1) has nothing to balance |
| Host-staged AR (SM kernel, PCIe via pinned host) | 25-31 GB/s, latency unknown | **a = 25-27 us fixed + 41 us/MB (24 GB/s)**, fit on decode-step calls (R=4: 29 us, R=16: 29.6, R=32: 37.9, R=64: 50.8). ±20% | the fixed latency is half of a decode AR |
| DMA (copy-engine) memcpy | — | H2D 3.3 us + 43.3 us/MB, D2H 2.9 us + 41.6 us/MB | the copy engines have ~3 us latency, vs ~25 us for the SM AR protocol |

Observed vs. what the framework assumed: the framework's ridge rule assumes GEMMs at DRAM BW below the
ridge. Here decode GEMMs reach only 60-70% of DRAM cold. The champion closes the gap with the AR's
L2-prefetch blocks: in situ, gate_up runs at an apparent 1.09 TB/s and qkvz at 34.7 us vs. 70 cold.
So DRAM time and link time are coupled through the AR's prefetch. A napkin that prices "weights at
BW" and "ARs at link speed" separately will double-count the overlap.

## 2. Decode step (verify step at 15 x 4 = 60 rows + 3 drafter passes)

Napkin uses: DRAM 957 GB/s for all streamed bytes; AR = 25 us + bytes/24 GB/s, with bytes = rows x 10 KB x 0.78 (champion's lossless compression);
small kernels priced at measured launch-bound times where the napkin has nothing better.

| Component | Napkin (bytes or count -> us) | Measured (median decode step, GPU 0, node trace) | Ratio |
|---|---|---|---|
| Body GEMMs, 64 layers | 12.18 GB -> **12.7 ms** | 12.47 ms (gate_up 5.31, out+down 4.60, qkvz 1.67, attn qkv 0.49, ba ~0.4) | 0.98 |
| GDN recurrent (state read 15 x 1.57 MB x 48 = 1.13 GB, write ~1/2 via lazy checkpoint) | ~1.7 GB -> 1.8 ms | 1.76 (replay kernel) + 0.19 conv + 0.07 bookkeeping = 2.02 | 1.12 |
| All-reduces, target (130 x [25 + 0.48 MB/24 GB/s = 20] us) | 130 x 45 = **5.9 ms** | **7.29 ms** (56 us avg; includes L2-prefetch blocks' issue time) | 1.24 |
| Attention (16 layers, ~500 ctx, fp8 KV) | ~0.15 GB -> 0.16 ms + launches | 0.43 ms | launch-bound |
| Norms, quant, silu-mul (~320 launches) | ~0.3 ms (bytes) | 0.92 ms | launch-bound |
| Target lm_head (0.57 GB 7-bit plane + gather + rescoring) | 0.6 ms | 0.80 (screen 0.65) + ~0.16 rejection/bookkeeping = 0.96 | 1.6 |
| Drafter, 3 passes: MTP weights 0.21 GB x 3, draft screen, 11 ARs, 5 all-gathers | 0.66 + 0.15 + 11 x 30 us = 1.1 ms | 1.76 ms | 1.6 |
| GPU idle inside the step | 0 | 0.16-0.22 ms | |
| **Sum** | **~22.0 ms** (12.7 + 1.8 + 5.9 + 0.16 + 0.3 + 0.6 + 1.1 + launch terms ~0.4 ... with the double-count of prefetch removed: see below) | **24.6 ms** (graph trace period, median of 550 steps) | **0.89** |

Overlap correction: if the AR's prefetch overlap is credited (in situ GEMM 12.47 vs. 12.7 at full DRAM, and
vs. ~20 ms cold), the napkin needs a term for "GEMMs at 60-70% of DRAM except what was prefetched". Fit:
the 130 ARs move ~1.6 ms of GEMM-weight DRAM traffic into L2. Per layer: gate_up apparent 82 us vs.
93 us at DRAM BW, qkvz 35 vs. 44. The rest of each AR's 56 us is link time with DRAM idle.

**Decode step model (used in 4-knobs):**
`T_dec ≈ 12.5 (GEMMs, in situ) + 2.0 (GDN) + 7.3 (target ARs) + 1.4 (attention + norms + quant) + 1.0 (lm_head + rejection) + 1.8 (drafter) + 0.2 idle ≈ 26.2` vs. 24.6 measured
(over by 6%, because kernels of different streams overlap a little in the node trace). Use the measured
components directly as the baseline for gains. Napkin gains are priced against these per-component times.

Per-row marginal costs in a decode-sized step (60 -> 61+ rows), from the microbenchmarks:
- GEMMs: cold bs_gemm sum per GDN layer is 320 us at M=64, 357 at M=128 and 474 at M=256. That is **~0.6-0.8 us per row per layer, ~37-51 us per row per step** below and around the ridge.
- ARs: 130 x 10 KB x 0.78 / 24 GB/s = **~43 us per row per step**. This is the dominant marginal cost below the ridge.
- GDN/attention: ~2 us per row for state-update rows (the replay kernel is per request; extra rows only add per-token recurrence work).

## 3. Mixed (prefill-containing) steps

| Source | Fit |
|---|---|
| graph trace, all mixed steps (n=80) | period = 7.8 ms + 149 us x (P+D) |
| node trace | 14.7 ms + 154 us x (P+D) |
| bs_gemm time in mixed steps | 10.9 ms + 95.9 us x (P+D), r2 0.98: GEMMs are ~62% of the per-row slope |
| in-situ bs_gemm rate | 152 TF (<256 tokens), 201 (256-767), 222 (>=768): 67-78% of the microbench at the same M |

Napkin per prefill row in a compute-bound step: body FLOPs 24.4 GFLOP per rank / 285 TF = **86 us** (GEMM floor).
Add the AR bytes: 43 us per row if exposed, ~0 if hidden under GEMMs by the champion's DMA split chain and waves.
Add the GDN chunked core and attention, ~5-10 us. Napkin: 90-140 us per row. Measured slope: ~150 us per row. Ratio ~0.6-0.9: the in-situ GEMM efficiency (§1) explains most of the gap.

Per-bin view from the graph trace, 96 prompts. Per prefill row = (period - 24.6 x D/60)/P:
| P bin | n | median P | median period | per prefill row |
|---|---|---|---|---|
| 1-63 | 13 | 60 | 31.8 ms | 147 us (fixed mixed-step overhead ~7 ms dominates) |
| 64-127 | 20 | 77 | 31.4 | 99 |
| 128-191 | 7 | 152 | 31.5 | **45** |
| 192-255 | 4 | 222 | 48.1 | 113 |
| 256-383 | 6 | 331 | 56.8 | 97 |
| 384-767 | 20 | 550 | 57.7 | **60** |
| >=768 (waves) | 10 | 946 | 184.7 | **171** |
Observed: a mixed step costs a ~7 ms fixed overhead over a decode step (probably eager or piecewise pieces and the GDN prefill path). Steps at >=768 rows cost ~3x more per row than steps at 384-767. The profile is 96 prompts: start and end bursts are over-represented. §4 gives the real distribution.

## 4. Whole trial (400 prompts)

Job 494 (`agents/leader-58.w1/calib/steplog.md`): champion, 400 prompts, c=16, no profiler, T37_STEPLOG.
- Throughput: 1225.2 tok/s (official 1212.4).
- Tokens: 90,632 generated, 104,526 input.
- Acceptance length: 2.688.
- Window: 2,393 steps in 75.9 s. Durations are host-side: the next step's timestamp minus this one's.

| Step type | steps | sum of durations | median | median-based time (steps x median) |
|---|---|---|---|---|
| decode-only, D=61-64 (full batch) | 1321 | 39.9 s | 24.8 ms | 32.8 s |
| decode-only, D<=60 | 723 | 19.5 s | 19-24.9 (19.0 at D=4, 20.7 at D=28) | ~16.9 s |
| mixed, P 1-191 | 184 | 5.6 s | **24.5 ms (= a decode step)** | 4.5 s |
| mixed, P 192-767 (DMA split chain, >= 256 rows) | 132 | 6.5 s | 38.8-41.6 | 5.3 s |
| mixed, P >= 768 (waves) | 33 | 4.4 s | 125 | 4.1 s |
| **total** | 2393 | **75.9 s** | | **63.6 s** |

Observed:
- Small prefills (P <= 191, total rows <= 255) ride on decode steps at no measurable extra step time (median 24.5 vs 24.7 ms). The marginal-row napkin of §2 predicts +37-51 us per row of GEMM + 43 us per row of AR, so +6-15 ms at 64-191 rows. **Napkin and measurement disagree.** Possible reasons: host-side attribution (the extra GPU time shows up in the next step's duration); or the GEMM/AR marginal costs are lower than the microbenchmarks suggest. Unresolved; GPU-side durations are requested (calib/idle.md).
- Medium prefills (192-767 P) cost +14-17 ms per step. Big ones (>= 768, 33 steps, spread across the whole run, not a start burst: ShareGPT has long prompts) cost ~100 ms extra each, ~115 us per prefill row.
- **~12.3 s of the 75.9 s (16%) is not explained by step-size medians.** The p90 decode-only step is 45.7 ms, the mean 29.1. The tail is either GPU work that spilled from the previous step (host-side attribution) or GPU idle (the graph trace had 1.6 s of idle in 14.3 s of decode windows, with 86 steps idle > 1 ms, p99 66 ms). Being resolved in calib/idle.md.
- The end-of-run tail (D <= 36: ~237 steps, ~5 s) runs at 19-20.5 ms per step for 1-9 requests. The step cost floor at 1 request is 19 ms (weights + ARs), so these steps produce ~4x fewer tokens per ms than full ones.

**Whole-trial napkin (seconds per trial):** full-batch decode 32.8 + partial-batch decode 16.9 + mixed 13.9 + unexplained tail 12.3 = 75.9.
Priced in 4-knobs.md as:
- decode-step improvements scale ~66 s (all decode-like steps incl. small mixed, median-based 54 s + their share of the tail);
- prefill-path improvements scale ~10-14 s;
- tail/idle improvements scale up to ~12 s.

### 4b. Correction: GPU-side times (calib/idle.md, graph trace of job 485, no new GPU work)

The host-side step durations above are **wrong per step**. With async scheduling the host runs ahead, and a step's GPU work lands in the next steps' host intervals. GPU-side facts:
- **Whole-window utilisation:** GPU0 98.4%, GPU1 98.0%. Idle 1.6-2%, of which 64% is kernel-to-kernel gaps under 0.1 ms. There are no idle gaps of 0.5 ms or more in decode-only steps. The 11 gaps in mixed steps (17 ms total) are mostly start-up one-offs (cudaFree, module load). **The "12 s unexplained tail" is not idle: it is mixed-step GPU work.**
- **Mixed-step GPU span** (first start to last end of the step's work, GPU1 medians):

| P bin | GPU span | host period |
|---|---|---|
| 1-63 | 43 ms | 32 |
| 64-127 | 47 | 31 |
| 128-191 | 60 | 32 |
| 192-255 | 74 | 48-54 |
| 256-383 | 89 | 53-57 |
| 384-767 | 115 | 55-58 |
| >=768 | 195 | 171-185 |

- **Decode-only step:** 24.5 ms GPU.

**Mixed-step model (GPU side):** `T_mix(P) ≈ 37 ms + ~0.15-0.165 ms x P`.
That is a decode step (24.5) plus a **fixed ~12.5 ms per mixed step** plus **~150-165 us per prefill row**.
Napkin per prefill row:
- GEMM floor 86 us at the 285 TF microbench, or 110 us at the 222 TF in situ;
- AR bytes 43 us if exposed;
- GDN chunked core and attention ~5-10 us.
That gives 100-160 us. **Ratio measured/napkin ≈ 1.0-1.5.** The 12.5 ms fixed overhead has no napkin term; its decomposition is pending (calib/mixed_vs_decode.md).

**Whole-trial GPU-side napkin** (400 prompts, 2044 decode-only + 349 mixed steps, 104.7K prefill rows):

| Part | Formula | Seconds | Share |
|---|---|---|---|
| decode-only steps | 2044 x 24.5 ms (partial-batch steps are a bit cheaper, ~19-24 ms) | ~48-50 | 64% |
| decode-equivalent part of mixed steps | 349 x 24.5 | 8.6 | 11% |
| mixed-step fixed overhead | 349 x ~12.5 | 4.4 | 6% |
| prefill rows | 104.7K x ~150 us | 15.7 | 21% |
| idle | 1.6-2% | ~1.4 | 2% |
| **sum** | | **~78-80** | vs 75.9 s measured (ratio 1.03-1.05) |

This table, not the host-side one, prices the candidates in 4-knobs.md. Corrections to 4-knobs' references:
- decode-step gains scale ~57 s (decode-only + decode-equivalent part of mixed steps), so 1 ms per decode step ≈ 2.4 s ≈ +3.1%;
- prefill-row gains scale ~16 s;
- mixed-step fixed overhead ~4.4 s;
- idle ~1.4 s.
