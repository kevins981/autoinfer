# leader-2 (T2) local KB
## State (2026-11-14)
- Sweep done: reports/T2-sweep/1-6, reports/T2-sweep.md, proposals/T2-map-1.md (accepted, applied).
- Split with T1: T1 = custom host AR, AR/GEMM overlap (W1), draft flags/depth, target local argmax.
  T2 = W8A8 GEMMs (decode+prefill), mixed-step lockstep (W4b), L2 prefetch during AR (W2), GDN DeltaLog, Marlin eff.
- Worker leader-2.w1 (ws t2-ar). Jobs: 2 (NCCL table), 8 (PCIe primitives). Task 3 = W8A8 vs Marlin microbench.
- ws t2-steplog: per-step GPU event timing (env T2_STEPLOG) in gpu_worker.py, measurement only.
## Facts
- Decode step 38.2 ms: Marlin 18.5, AR 7.75 (60us each), lm_heads 4.06, AG 1.46, GDN 2.95, idle 1.0.
- Mixed step 147 ms (nsys): Marlin 59, AR 62.5 (half = rank skew 150us median), idle 24.5.
- GDN MTP kernel writes 3 states/req/layer (csrc/gdn/fused_gdn_decode_kernel.cu ~L320).
- vLLM options: speculative_config.use_local_argmax_reduction, draft_lm_head_quantization fp8.
- Kernels: TritonFp8BlockScaledMMKernel, CutlassFp8BlockScaledMMKernel in vllm/model_executor/kernels/linear/scaled_mm/.
## Process change (human via manager, 2026-11-14)
- Shared map.md dropped. Each team's map = reports/T2-sweep/ (+ update-<n>.md at each resync and closing:
  tested results, new candidates/dimensions, changed verdicts with evidence, current ranking). No more map proposals.
- My 5-compare.md compared against the (then) shared map + T1's sweep; future comparisons: other teams' sweep folders.
## Results (2026-11-14)
- T2-001: capture sizes to 512 on T1-004: +6.42% ABBA (job 27). T2-002 = T1-005 + T2-001: official 757.6 (+19.1% BASE).
- T2-003: Triton W8A8 on BASE: no gain (mixed steps were eager). Re-test on T2-002: worker, ws t2-w8a8c vs t2-champ.
- T1 info: AR/GEMM overlap dead (T1-007: Marlin and AR kernel don't co-run on SMs) -> threatens my W2 (L2 prefetch
  concurrent with AR): check how many SMs hostmem AR / NCCL kernels use and whether a prefetch kernel can co-run.
- T1 testing TRITON_ATTN (removes 2 host syncs, ~1.2-1.9 ms/step idle).
- T2-004 (W8A8 on T2-002): ABBA +4.73% (job 30), GSM8K-250 A/B ref 93.6 / fp8 92.4 (job 32), 128K pass; official eval requested.
- T2-005: L2 prefetch during AR: microbench negative (job 33), 0-5 us/layer.
- Chunking prefills to <=512 rows: not worth it: steps >512 rows (14% of time) already GPU-bound
  (231 ms med at 763 rows vs 233 ms graphed estimate; steplog3).
- Pending: worker task 7 = ncu of GDN decode kernel (DeltaLog decision).
- T2-004 official 788.8 (+0.3% vs T1-009 786.4: not promoted). T2-006 = T2-004 + TRITON_ATTN: official queued.
- GDN ncu (job 36): read 26 MB, SM->L2 writes 75.8 MB (3 states), DRAM writes in-window 35 MB, 75 us, 82% DRAM peak.
  DeltaLog napkin 1.3-3 ms/step. Worker task 8: build in ws t2-gdn (from T2-004), correctness test, no A/B yet.
- T2-006 champion 819.9 (credited T2+T1). Resync 1: +60 GPU-min. update-1.md written.
- Ruling: combos may carry over the FP8-act GSM8K A/B if additions are exact/fp-reordering and don't touch FP8 paths.
- DeltaLog stage 1: kernels + bitwise test PASS (job 40). Next: timing (48-layer emulation) + negative control.
- T2-007: VLLM_USE_BREAKABLE_CUDAGRAPH=0 screen negative -7.5% (job 46): model has no compile path; loses mixed graphs.
- nsys T2-006 (agents/leader-2/prof_t2006): decode 31.7 ms: Triton GEMM 16.7, hostmem AR 5.9, GDN 3.0, draft lm_head 1.44,
  target lm_head 1.58, [WRONG: those NCCL ARs were mixed-step kernels caught by CPU-NVTX windows; per T1 decode NCCL = SendRecv 0.36-0.49 ms bytes], SendRecv 0.49. Mixed 112 ms: GEMM 53.6, NCCL AR 31.8.
- gate_up in GDN layers 137.7 us vs attn layers 113.6 us (same steps): GDN dirty-write drain ~1.15 ms/step -> DeltaLog
  total napkin ~2-2.5 ms/step.
- Usage-limit stop ~19:20; resumed worker (stage 2 of DeltaLog). T2 balance 79.9.
- Next candidate after DeltaLog: fuse RMSNorm+group quant (ops.rms_norm_per_block_quant) and silu_and_mul+quant
  (ops.silu_and_mul_per_block_quant) by hand (no torch.compile here). Napkin ~0.6 ms/decode step (quant kernel) ~1.5%.
  Touches FP8 path -> needs own GSM8K-250 A/B.

## 2026-11-14 DeltaLog result
- T2-008 = T1-014 + DeltaLog (default on). ABBA job 65: +3.91%. OFFICIAL #1: 880.1 tok/s (+4.4% vs T1-014), GSM8K 94.4%, all gates pass.
- Napkin was ~7%; got ~4%. Materialize kernels in mixed/boundary steps probably eat part of it; TPOT p99 30 -> 34 ms.
- T1 closed; T3 opened. Next: fused RMSNorm+group-quant (worker task 9b), decode-M config tuning.
- T2-009 (T3-003 + fused norm/SiLU+quant): ABBA +1.29% (job 102), OFFICIAL 906.9 (+1.06%), GSM8K 92.8. Resync requested, update-2.md written. Balance 20.5.
- T3-003 = T2-008 + L2 prefetch in the hostmem AR (+1.97%), which overturned my T2-005 verdict.
- Worker task 10: decode-M GEMM microbench (cold and with 40 MB prefetched).
- LESSON: T2-009's fused norm+quant silently disabled T1-013's row-split AR overlap (rowsplit_ok() False). Found by T3 (job 117); T3 is fixing it (ws t3-rs). From now on, check the feature log lines of earlier wins ("Row-split TP all-reduce pipelining active", "fused ... active", DeltaLog) in every server log before an ABBA.
- T2-010: GEMM split-K/pf configs, smoke inconclusive, dropped. T2-011 = T3-006 + reduced-vocab draft head K=151643: ABBA +3.14% (job 116); official running.
- Spar T2-1 (sparring/T2-1.md):
  - "k=2 optimal" is TO BE TESTED: T1's k=3 losses (T1-003, T1-010) predate W8A8, DeltaLog, the reduced-vocab draft head and the faster AR. ABBA k=3 job queued (abba_k3.sh). DeltaLog caps at 2 logged rows, so k=3 falls back to full-state writes (worst case).
  - "Mixed-step GEMMs at ceiling" is ASSUMED (65-79% of 330 TFLOPS nominal). Needs an independent bound (sustained clock, MMA rate, scale-FMA overhead). Worker writing the note.
- T2-012 (AR release/acquire on T3-009): ABBA +1.08% (job 129); official running (job 131).
- CLOSING: T2-012 vs T3-009 ABBA was mixed (+0.54%). k=3 screen -11.7% (T2-013). Wrote update-3.md and T2-writeup.md. Recommending closure with 17 GPU-min left.
