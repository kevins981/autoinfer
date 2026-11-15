team:         T2
builds_on:    T3-003 (champion)
performance:  "OFFICIAL #1: 906.9 tok/s (+1.06% vs T3-003 897.4, +42.6% vs BASE), GSM8K 92.8%, all gates pass. Unofficial: +1.29% ABBA job 102"
summary:
  hypothesis: With no torch.compile (breakable CUDA graphs force CompilationMode.NONE), each W8A8 GEMM input pays a
              separate norm (FlashInfer fused_add_rmsnorm ~2.75 us) or SiLU-mul (~2.6 us) kernel plus a
              per_token_group_quant (~2.4 us). Fusing the quant into the producer saves about one launch per site:
              ~0.5 ms per decode step (~1.5%). It may also evict fewer T3-003-prefetched weight lines.
  change:     (worker leader-2.w1, ws t2-nq3)
              - New Triton add-RMSNorm + group-128 FP8 quant (models/qwen3_5/nvidia/ops/fused_norm_quant.py), using
                inline PTX cvt.rn.satfinite.e4m3x2.f32. Triton's own fp8 cast double-rounds via fp16 and was rejected.
              - Wired via Qwen3_5RMSNorm.forward_fused for the in_proj_qkvz/qkv_proj and gate_up consumers.
                The bf16 y is still written for in_proj_ba.
              - New Triton SiLU-mul + group quant for down_proj (fusion/fused_act_quant.py).
              - TritonFp8BlockScaledMMKernel accepts a pre-quantized activation (scaled_mm/triton.py, fp8.py).
              - mamba_utils _deltalog_heads raises if the gate is on but the state layout doesn't match.
              - Gate VLLM_T2_FUSED_NQ (default 1; 0 = T3-003 path).
  results:    Observed:
              - SiLU site: bitwise equal to silu_and_mul + per_token_group_quant_fp8 for M 1..1000 (worker job 98).
                The existing C op silu_and_mul_per_block_quant is NOT bitwise (skips the bf16 rounding of silu)
                and was not used.
              - Norm site: residual and scales bitwise. y differs from FlashInfer in ~1e-5 of elements by 1 bf16
                ulp (fp32 variance reduction order), flipping ~1 fp8 code per several 100k (job 92).
              - Smoke job 100: +1.35% in 1 pair.
              - ABBA job 102 (same workspace, A gate 0): +1.29%, all B > all A.
                Acceptance 2.345/2.352 (A) vs 2.349/2.341 (B).
                TPOT p99 31.2/33.4 (A) vs 30.3/32.4 ms (B).
              - Same-job GSM8K-250: A 93.2%, B 93.6%.
              Napkin was ~0.5 ms/step (~1.5%); measured ~1.3%.
  conclusion: Interpretation: small and consistent; matches the launch-count napkin.
validity:     (leader-2) Under the manager's fused bf16-op + FP8-quant ruling (task_definition.md):
              - Quant stage bitwise identical. The only other difference is the fp32 reduction order inside RMSNorm.
              - Same-job GSM8K-250 on both arms: 93.6 vs 93.2.
              - Quant scheme, scales and precision unchanged; no communication change; no memory change
                (no 128K test needed, the official gate runs it).
              - Not benchmark-specific. T3-003's prefetch hint logic is untouched.

official #1 (2026-11-14T23:26): 906.9 tok/s at c=16 (mean of 2 trials: 904.8 / 909.0, CV 0.3%) (+42.6% vs BASE), GSM8K 92.8%, MTP acceptance 2.343, TTFT p50 160 ms, TPOT p50 16.2 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
