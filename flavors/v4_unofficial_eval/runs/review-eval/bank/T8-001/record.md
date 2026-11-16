team:         T8
builds_on:    T5-001 (microbenchmark only; no engine change filed)
performance:  not measured by the harness. Ruled NOT ALLOWED by the manager (2026-11-16, task_definition.md) before any server run.
summary:
  hypothesis: mixed prefill+decode steps (~29 s of the champion's 101 s) are compute-bound in the W8A8 GEMMs. On sm_89 the FP8 mma also
              has an fp16-accumulate form at ~2x the fp32-accumulate peak. I read the W8A8 rule ("the FP8 tensor cores' own accumulator
              is fine within a block") as allowing it inside each 128-K block, with fp32 scaling and accumulation across blocks.
  change:     microbenchmark notes/f16acc.py (standalone Triton block-scaled kernel, random data, fp32 vs fp16 out_dtype in tl.dot per
              128-K block). An engine patch (notes/f16acc_kernel_unmeasured.diff: F16_ACC for M >= 256 in _w8a8_triton_block_scaled_mm,
              with a tile-level fallback to fp32 accumulation if any fp16 partial is non-finite or > 65504) was written but reverted
              unbuilt and unmeasured after the ruling.
  results:    Observed (GPU job 124, 0.2 GPU-min, GPU 0, Triton do_bench):
              gate_up M=512 N=17408 K=5120: best fp32-acc 378 us (241 TF, cfg 64x128x128 w4 s3 = the champion's config) vs best
                fp16-acc 308 us (296 TF): 1.23x; same config 1.23x; other configs 1.34-1.44x.
              down M=512 N=5120 K=8704: 220 vs 203 us best-vs-best (1.09x).
              gate_up M=2048: 1334 vs 1087 us best-vs-best (1.23x); 274 vs 336 TF.
              Random-data max abs diff vs fp32 acc 2.4e-4 (rel 4e-3 of max output, about bf16 output rounding); no inf on this data.
  conclusion: Interpretation: (1) fp16 accumulation gives ~1.1-1.25x on prefill GEMMs, best config against best config, not the 2x the
              spec suggests. Napkin overall: about +2-3% if GEMMs are ~60% of mixed-step time. (2) fp32-accumulate W8A8 already reaches
              240-275 TFLOP/s at M>=512, well above SPEED_OF_LIGHT's 165 TFLOP/s "fp32-acc" figure. So prefill GEMMs are not far from
              peak, and the mixed-step excess other teams measured (T4: 72-122 ms per small mixed step) is mostly not GEMM compute.
              (3) Ruled out by the manager: the fp16 accumulator is a lower-precision accumulator ("nothing coarser or lower"), not fp
              reordering. Do not rebuild unless the human overrules.
validity:     leader-8: the idea was disallowed before any server run; nothing here is a champion candidate. The microbenchmark uses
              random data only.
