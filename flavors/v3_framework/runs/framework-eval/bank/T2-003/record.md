team:         T2
builds_on:    BASE
performance:  "unofficial: within noise (640.4 vs BASE 639.3 tok/s, 200 prompts, cross-job: T2 jobs 24 vs 10)"
summary:
  hypothesis: W8A8 (checkpoint's own scheme: dynamic per-token-group-128 FP8 activations x 128x128 block FP8 weights,
              Triton block-scaled GEMM) beats Marlin (bf16 MMA on dequantized FP8) at M>=128 by 1.2-1.6x and ties at
              decode M~45 with 4090-tuned configs (microbench T2 job 11; tuning job 16).
  change:     force TritonFp8BlockScaledMMKernel for FP8 block linears on sm89 (env VLLM_FP8_BLOCK_PREFER_TRITON, default
              on for sm89) + 5 tuned config JSONs (NVIDIA_GeForce_RTX_4090) for the body/MTP shapes. lm_head untouched.
              Worker leader-2.w1, ws t2-w8a8.
  results:    Observed (job 24): 640.4 tok/s, acceptance 2.346, TTFT p50/p99 208/1532 ms, TPOT p99 42.4 ms, GSM8K-50 88%
              (= BASE's 44/50). Microbench (job 11): at M=45 Triton+quant 0.90-1.03x Marlin time; at M=260 0.62-0.90x.
  conclusion: Interpretation: no gain because mixed steps (where the GEMM gain is) were eager / CPU-launch-bound in BASE
              (see T2-001); decode ties. To be re-tested on top of T2-001 (mixed steps graphed).
validity:     Allowed by task_definition (checkpoint W8A8 scheme): dynamic per-token-group-128 e4m3 activations, fp32
              accumulation across K-blocks (accumulator += dot * a_s * b_s), no static/per-tensor scales; norms,
              attention/GDN, KV, comms, lm_head unchanged. Requires the same-job GSM8K-250 A/B vs bf16 reference (within
              2.0 points) before any official eval: not yet run.
