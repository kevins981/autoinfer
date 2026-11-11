# T13-001: premise microbench for "producer-fused AR" (leader-13.w1)

Job 113 (final, GPU0 then GPU1, 100 reps/cond, medians); smoke jobs 111, 112 (GPU0 only, same numbers). GPU time used: ~0.8 GPU-min of 6.
Files: agents/leader-13.w1/t001/{mb.cu, bench.py, job.sh, result_gpu{0,1}.json, log_gpu{0,1}.txt}. Engine code untouched.

## Method (what was measured)
- Real shapes (Qwen3.8-27B-FP8, TP=2, block-FP8 g=128, random weights, repacked with the engine's gptq_marlin_repack, op `torch.ops._C.marlin_gemm` from the workspace build = champion tile logic, fp32 reduce on), M=48:
  - out_proj (GDN) and o_proj (attn) are the SAME shape: K=3072, N=5120 -> measured once.
  - down: K=8704, N=5120. gate_up: K=5120, N=17408 (reference).
- Payload 383,872 B (48x5120x2x0.781). Host buffer: /dev/shm file, mmap, first-touch, cudaHostRegister.
- Timing: GEMM in a CUDA graph bracketed by two 1-thread %globaltimer kernels (gemm_us = t1-t0, includes ~1.2us of kernel-boundary gaps; profiler shows Marlin itself = gemm_us - ~1.3). Side kernels (st.global.cs.v4 writes / ld.global.cv.v4 reads, 256 thr/CTA) are launched on other streams, spin on a device flag that the t0 kernel bumps, so they start 0.4us after t0. Write time = first CTA start .. last CTA after __threadfence_system.
- COLD = 256 MB memset before each rep (weights come from HBM, like the 64-layer cycle). WARM = no flush (weights in L2).
- Placement: GPU1 process bound to node 0 and verified (numa_maps N0=1024 pages). **GPU0 process in job 113 did NOT bind (self_bind returned None; pages on N1)**; the smoke jobs' GPU0 runs were bound to N0 (verified) and gave the same numbers (wr8 ~22.7 GB/s, GEMM +8us), so placement made no visible difference here.

## A. Interference split (COLD, us; GPU0 / GPU1 in job 113; d = GEMM slowdown vs GEMM alone)
Side kernels alone (no GEMM, GPU0): wr8 22.3 GB/s (17.2 us), wr16 23.4 GB/s, rd8 24.4 GB/s (15.7 us), rd16 23.9, wr8+rd8 concurrently 15.9 + 13.8 GB/s, D2H copy-engine 383 KB: 24.7 us.

| condition | out_proj/o_proj (alone 35.7 / 33.1) | down (80.0 / 76.4) | gate_up (144.9 / 140.7) |
|---|---|---|---|
| +wr8: d ; write GB/s | +9.2 (+26%) ; 22.4 / +9.5 (+29%) ; 21.5 | +8.7 (+11%) ; 22.2 / +8.9 (+12%) ; 22.5 | +9.2 (+6%) ; 22.0 / +10.3 (+7%) ; 21.0 |
| +wr16 | +9.2 ; 22.0 / +10.1 ; 21.3 | +9.4 ; 21.5 / +9.4 ; 20.9 | +10.2 ; 21.3 / +10.6 ; 22.0 |
| +rd8: d ; read GB/s | +7.1 ; 17.8 / +7.2 ; 17.4 | +6.6 ; 16.6 / +7.1 ; 15.0 | +7.9 ; 15.4 / +7.6 ; 14.8 |
| +rd16 | +4.4 ; 22.9 / +4.5 ; 22.8 | +4.4 ; 22.1 / +4.0 ; 21.6 | +5.3 ; 22.2 / +4.6 ; 22.1 |
| +wr8 & rd8 together | +16.0 / +15.7 | +14.5 / +14.5 | +15.2 / +16.0 |
| +D2H copy engine | +0.5 (+2%) / +0.8 | +0.7 / +0.4 | +0.6 / +0.6 |
| serial: wr8 after GEMM (total d) | +16.0 / +16.2 | +16.8 / +16.8 | +16.7 / +16.5 |
| wr1 CTA: d ; GB/s | +6.1 ; 19.6 / +6.4 ; 18.3 | +0.3 ; 16.2 / -0.1 ; 15.9 | -0.4 ; 16.2 / +0.2 ; 15.8 |
| wr2 | +5.9 ; 21.5 / +5.4 ; 22.5 | +1.3 ; 20.7 / +1.6 ; 21.3 | +2.2 ; 20.9 / +3.7 ; 22.0 |
| wr4 | +7.6 ; 23.0 / +8.2 ; 23.2 | +6.5 ; 22.2 / +6.6 ; 22.7 | +7.1 ; 22.6 / +8.6 ; 22.7 |
| wr32 / wr128 | +10.5 / +13.2 (GPU1 +10.5 / +13.5) | +10.1 / +13.2 | +10.7 / +14.1 |
| wr8 of 96 KB / 192 KB (quarter / half) | +0.2 / +2.6 (GPU1 +0.9 / +2.5) | +0.4 / +2.4 | +0.4 / +2.8 |
| controls: 0-byte / 4 KB writer, 8-16-128 CTAs | -0.7..+0.8 (noise) | ~0 | ~0 |
WARM (no flush, weights L2-resident) rows: GEMM alone_warm = 21.9 / 46.4 / 108.0 us (out_proj / down / gate_up, GPU0), but GEMM + wr8_warm (or +rd8_warm) = 36.4 / 61.3 / 120.8 (GPU1 the same pattern). Observed only; UNEXPLAINED (the warm benefit mostly vanishes when a host writer/reader runs beside it; possibly the sysmem traffic or the extra kernels disturb L2 residency, or an artifact of the gated launch). I did not follow it up: do not use the WARM rows as evidence either way.

Gate A2 (writes-only, GEMM slowdown <= 5% and write rate >= 18 GB/s):
- Write rate beside the GEMM is NOT degraded: 21-22.5 GB/s at 8/16 CTAs (alone 22.3-23.4). Passes the rate half everywhere (>= 20.7 GB/s for >= 2 CTAs).
- GEMM slowdown FAILS at 8/16 CTAs for every layer: +9 us (~constant, independent of GEMM length): +26-29% out_proj, +11-12% down, +6-8% gate_up.
- Passes only with 1-2 writer CTAs on down / gate_up (+0..+3 us = 0-3%, 16-22 GB/s: wr2 20.7-22.0 GB/s, d +1.3..+3.7 us (2% / 2-3%)); never on out_proj/o_proj (+5.4-6.4 us even with 1-2 CTAs).

## B. Marlin output-tile completion timeline (COLD, N=5120 split into 80 chunks of 64 cols; a chunk is 'done' when 2-4 probe elements (rows 0 and 47, first/last col) leave the 0xFFFF sentinel)
**The poller (1 CTA) itself perturbs the GEMM** by +8..9 us (out_proj, +24-26%) and +9.4-12 us (down, +12-15%), independent of poll intensity (np4 no sleep / np2 +300 ns sleep / np1 +1 us sleep give the same). Fails the 'few %' requirement, so fractions are of the PERTURBED kernel.

| (fractions of t1-t0 incl. poller perturbation; GPU0 / GPU1) | out_proj/o_proj | down |
|---|---|---|
| kernel duration with poller (alone) | 44.8-45.0 (35.7) / 41.1 (33.1) | 91.9 (80.0) / 85.9 (76.4) |
| first / median / last chunk done | 0.58 / 0.69 / 0.99 ; 0.61 / 0.72 / 0.99 | 0.75 / 0.82 / 0.996 ; 0.79 / 0.845 / 0.997 |
| chunks done by 25% / 50% | 0% / 0% (both GPUs) | 0% / 0% |
| by 70% | 55-63% (GPU0) ; 29-35% (GPU1) | 0% / 0% |
| by 75% | 91-94% ; 74-85% | 0.8% / 0% |
| by 90% | 98.8% ; 98.8% | 98.7% / 98% |
| median chunk done (absolute, us after t0) | 31.0-31.3 ; 29.4-29.7 | 75.2-75.5 ; 72.3-72.8 |

- All chunks except the LAST one (cols 5056-5119, and the one before) complete in a narrow window (spread ~0.60-0.74 for out_proj, 0.77-0.86 for down, with a period-5 chunk pattern); the last chunk completes at the kernel's end.
- Reading the absolute times against the UNPERTURBED kernel (35.7 / 80.0 us) gives median chunk done at ~87% (out_proj) and ~94% (down) of the kernel, i.e. only a ~4-5 us tail in both. That is an estimate (assumes the poller delays only the straggler block, as the 'last chunk at 0.99' suggests; the unperturbed timeline could not be observed).
- Tile width of the chosen Marlin config not determined (I probed 64-col chunks; did not instrument Marlin).

Gate B (>= 50% of tiles done before 70% of the GEMM): FAIL in the perturbed reading for down (0%) and GPU1 out_proj (29-35%); GPU0 out_proj marginally passes it (55-63%) only because the perturbed kernel is stretched; with the unperturbed-duration reading it fails everywhere (median ~87-94%).

## Observations kept short
1. Posted SM writes to pinned host memory run at the same ~22 GB/s beside a streaming Marlin as alone (no x1.45 on the writes themselves).
2. They slow the COLD GEMM by a ~constant ~9 us for 383 KB at >= 8 CTAs (6-29%), scaling with write volume: 96 KB ~0, 192 KB ~+2.5 us, 383 KB ~+9 us; 0-byte/4 KB side kernels cost ~0, so it is the write traffic, not CTA co-residency/smem. 1-2 writer CTAs cost 0-3 us on down/gate_up but ~6 us on out_proj.
3. Concurrent write+read (round trip) costs +15-16 us ~ the full serial write cost (+16 us); copy-engine D2H costs ~0.5 us (T8-001 confirmed).
4. Serial write after GEMM costs ~16 us; concurrent wr8 saves at most ~7 us of that (not 16) and only if the data exists, which B says it doesn't until the last ~5-13% of the kernel.

## Interpretation (mine)
- The idea's two premises both fail: (1) writes beside the GEMM are not free (A2 slowdown gate failed at 8/16 CTAs; works only at 1-2 CTAs on the longer GEMMs); (2) Marlin's output tiles are produced late and nearly together (K-split stripes finish in lockstep), so there is almost nothing to overlap with: a ~4-5 us tail (unperturbed estimate), vs ~17 us needed to push 383 KB at ~22 GB/s even with zero interference. Upper bound for the best case ~4-5 us per AR minus any interference, x128 ARs/step.
- A more promising reading of A: copy-engine paths cost ~0 beside the GEMM, and writes cost proportional to volume (96 KB ~free) -> smaller payload or writing only part (e.g. a quarter) alongside may be ~free, but I did not test a partial scheme.
- Doubts: WARM rows unreliable (see above); poller perturbation; GPU0 unbound in job 113; one job only (100 reps, medians; GPU0 vs GPU1 agree within ~1 us).
