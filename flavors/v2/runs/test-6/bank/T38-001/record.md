team:         T38
builds_on:    T37-002 (measurement only)
performance:  "measurement only: proposal T38-1 (stream the decode AR into its consumer GEMM) closed before building: its ceiling was already measured below 5% (T4-001)"
summary:
  hypothesis: the decode all-reduce (7 ms of a ~26 ms step) can stop being a barrier: deliver the reduced row in column
              chunks and let the next weight-streaming GEMM wait per K-chunk (1/rms moved to the epilogue per the
              2026-11-09 ruling), hiding ~45-65 us/layer (~3-4 ms/step, +8-12%).
  change:     none to the engine. Measurements: champion profile (prof_champ.sh), FP8 mma peak (mb/mma_peak.cu),
              f16-accumulate variant of bs_gemm_marlin (mb/bs16.cu, bench16.py), independent AR||GEMM overlap
              (ovl/ovl_test.py). All under agents/leader-38/.
  results:    Observed:
    - job 382 (nsys node trace, champion, 80 prompts; prof_champ_node/, scripts anat.py, gpusteps.py, gpusteps2.py):
      decode step 16 req x 4 rows: ~24-26 ms, GPU busy ~100%. Per step (GPU0): hostar_kernel 7.05 ms (141 calls),
      t31b 6.87, Marlin 6.79, GDN replay 1.81, target lm_head 1.03, drafter ~3 x 0.68 ms. Per GDN layer ~370 us:
      AR 55, gate_up(Marlin, 89 MB) 95-103, down(t31b) 53, AR 48-50, in_proj(t31b) 34.
      Mixed steps (GPU-defined, lm_head to lm_head): ~60 prompt tokens 42.5 ms vs decode 23.8 (+18.7);
      ~330 tokens 96-110 ms; bs_gemm 20.5 ms (M~120) / 61 ms (M~390, ~155 TFLOPS); mixed steps ~29% of GPU time.
    - job 383: mma.sync m16n8k32 e4m3: f32 accumulate 330 TFLOPS, f16 accumulate 659; bf16 165 (both GPUs).
    - job 386: bs_gemm_marlin standalone (GPU0, random data): f32-acc best configs 184-294 TFLOPS (M 128-2048);
      f16-acc variant only 1.09-1.48x faster AND overflows on ~100% of outputs with checkpoint-style scaling
      (e4m3 activations to 448 x block weights to 448). f16 in-block accumulation is not viable as is (question withdrawn).
    - job 388: champion HostStagedAllReduce (no prefetch) at 64 rows 61 us, independent bf16 GEMM over 89 MB 102 us:
      serial 167, concurrent 110.5 us; 128 rows: 218 vs 144. (Total time only; per-kernel stretch not separated.)
  conclusion: Interpretation: job 388 is consistent with T4-001 (pointed out by the manager): the AR co-runs with a
              DRAM-streaming GEMM but stretches to ~100 us, so it only hides behind gate_up; T4-001's ceiling arm
              (consumer not waiting at all) saved 5-8 us/AR in a 16-layer real chain. The dependent version can only
              be worse. Upside < 5% -> core idea wrong on this machine (PCIe AR traffic and HBM streaming contend;
              the champion's L2 prefetch already cashes ~18 us/AR). Not built.
validity:     measurement only; no served change.

Addendum (2026-11-10, proposal T38-2 closed without GPU): a self-distilled, recursively trained MTP head was already
measured by T3-001..T3-004 (reports/T3-writeup.md): offline proxy validated in-server; hard-label, LoRA/step-2 and KL
fine-tunes flat on held-out data; ~+0.35 pt a2 per data doubling. The riskiest assumption of T38-2 is shown false at
affordable data cost. No GPU spent on T38-2.
