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
