team:         T2
builds_on:    T1-042
performance:  "microbenchmark only (no server): measures decode/prefill co-running on green contexts"
summary:
  hypothesis: Prefill (tensor-bound) can run on a green-context SM partition while decode (memory-bound) keeps running, so the two overlap instead of being serialized in mixed steps.
  change:     T1-042 + MARLIN_SMS env override in csrc/quantization/marlin/marlin.cu (grid size; read at capture). Benchmarks: peak.py (job 1), mb2.py (job 6, by leader-2.w1).
  results:    Observed:
    - job 1: bf16 dense peak GPU0 144 / GPU1 158 TF; cuBLAS bf16 at M=256-320 model shapes 125-140 TF; green contexts work on sm_89 (gate_up M=4096: 32/64/96 SMs -> 35.8/71.5/106.5 TF).
    - job 6 (GPU 0, real FP8 Marlin, 16 layers x 4 GEMMs, distinct weights, CUDA graphs):
      decode chain M=48 alone 3.99 ms (767 GB/s). Green ctx with sms=N: 96 -> 4.10, 80 -> 4.41, 64 -> 5.29 ms. With default sms=128 on any partition: ~5.75 ms (2 waves).
      prefill chain M=256: full 12.74 ms (123 TF); ~1 TF/SM on partitions (32 SMs 34 TF, 48 SMs 50 TF, 64 SMs 67 TF).
      co-run, decode on plain stream (sms=128-f) + prefill on green ctx f: f=32 dec 4.08->5.25, pre 32 TF; f=48 dec 4.39->5.75, pre 45 TF; f=64 dec 5.31->6.49, pre 59 TF.
      green/green co-runs are much worse (decode 1.8-5x); plain streams with priorities serialize graph replays completely (15.93 = sum).
  conclusion: Interpretation: spatial overlap works for GEMMs when the decode runs on the plain stream with a Marlin grid sized to 128-f and the prefill sits on a green context: decode Marlin slows 1.28-1.31x (f=32-48) while prefill keeps ~90-95% of its partition rate. Pure-GEMM chains gain 8-18% over serialized; the real decode step has ~12 ms/25.5 of non-Marlin time (AR windows) that this bench doesn't include. One run each, no repeats.
validity:     measurement only, no served change. The MARLIN_SMS override only changes the grid partition (fp reduction order at most).
