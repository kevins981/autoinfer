team:         T2
builds_on:    T2-002 + T2-003
performance:  "official #1: 788.8 tok/s (+4.1% vs T2-002 757.6, +0.3% vs T1-009 786.4, +24.0% vs BASE), all gates pass; unofficial ABBA +4.73% vs T2-002 (job 30)"
summary:
  hypothesis: T2-003 (Triton W8A8 block-FP8 GEMMs, 4090-tuned configs) showed no gain on BASE because the GEMM saving is
              in mixed steps (M 128-512: Triton+quant 0.62-0.90x Marlin, job 11), which ran eagerly. With mixed steps
              graphed (T2-001/T2-002) the saving should become visible; decode (M~45) ties Marlin.
  change:     T2-002's code and flags + T2-003's change: FP8 block linears (body + MTP layer) use TritonFp8BlockScaledMMKernel
              on sm89 by default (env VLLM_FP8_BLOCK_PREFER_TRITON=0 restores Marlin), + 5 tuned Triton config JSONs
              (device NVIDIA_GeForce_RTX_4090; tuned on random weights/activations by shape, job 16). lm_head untouched.
  results:    Observed: ABBA (job 30, worker leader-2.w1, agents/leader-2.w1/out/abba_w8a8/): A (T2-002) 745.8/746.8,
              B 781.7/781.5 tok/s (+4.73%); acceptance A 2.352/2.348, B 2.338/2.341; p99 TTFT A 1252/1241, B 1034/1003 ms;
              p99 TPOT A 38.6/40.8, B 36.7/39.7 ms. No "Using default W8A8" warnings; graphs 1.42 GiB/GPU; KV unchanged.
              Contract GSM8K-250 A/B, same job, same code (job 32, agents/leader-2/gsm_ab/): bf16-activation ref (Marlin,
              PREFER_TRITON=0) 93.6% vs FP8 activations 92.4% (diff 1.2 <= 2.0 points). 128K request with B: pass (46.1 s).
  results+:   official #1 788.8 (789.1/788.5), GSM8K 94.0%, acceptance 2.338, TTFT p99 961 ms, TPOT p99 32.2 ms, 128K pass.
  conclusion: Interpretation: once mixed steps are CUDA-graphed, their faster W8A8 GEMMs show (~+4.7%); TTFT p99 improves.
validity:     Checkpoint's own W8A8 scheme (task_definition, 2026-11-09): dynamic per-token-group-128 e4m3 activation
              quant (per_token_group_quant_fp8, no static/per-tensor scales), against 128x128 block-scaled FP8 weights;
              Triton kernel accumulates fp32 across K-blocks (accumulator += dot * a_s * b_s). Norms, attention/GDN, KV,
              communication (bf16) and target lm_head unchanged. GSM8K-250 A/B within 2.0 points (above). Configs are by
              shape, tuned on random data, not on benchmark data. Generic default keyed to sm89 only.

official #1 (2026-11-14T18:40): 788.8 tok/s at c=16 (mean of 2 trials: 789.1 / 788.5, CV 0.1%) (+24.0% vs BASE), GSM8K 94.0%, MTP acceptance 2.338, TTFT p50 174 ms, TPOT p50 18.6 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
