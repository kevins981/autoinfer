team:         T45
builds_on:    T37-002 (+ T45-004's finding)
performance:  "official #1: 1173.4 tok/s (+2.7% vs T37-002 cross-job); unofficial: same-job ABBA +3.83% all B > all A (job 415); FP8 GSM8K A/B +1.6 pt (job 416)"
summary:
  hypothesis: At k=3 decode (16 req x 4 rows = M=64) the champion's gate_up stays on bf16-activation Marlin
              (T31_SKIP=17408x5120, W8A8 only above M=96) and sits on Marlin's M>48 step (KB94). T45-004's chain
              measured 104.1 vs 82.6 us/layer for T29's bs_gemm_marlin W8A8 (cfg 127). Est. +1.5-2.5% e2e.
  change:     7 lines in vllm/model_executor/layers/quantization/utils/t29_marlin_w8a8.py: if T45_GU_BS (env,
              default 1) and T45_GU_BS_MIN_M (48) < M <= T29_W8A8_MAX_MARLIN_M (96) and (N,K) == (17408,5120),
              _t29_marlin_fp8_linear returns w8a8_marlin(...): the existing per-128-group e4m3 activation quant
              (quant_perm) + bs_gemm_marlin with T29's config table (cfg 127 at M<=96), i.e. the same code the
              champion already runs for gate_up at M>96. Every other shape/M unchanged. T45_GU_BS=0 restores
              T37-002 exactly. serve_args = T37-002's.
  results:    Observed:
    - job 415 ABBA, 300 prompts, c=16, same workspace t45-gu, A = T45_GU_BS=0 (champion dispatch), B = default:
      A 1107.5 / 1116.5, B 1150.4 / 1158.7 tok/s -> all B > all A, +3.83%. Acceptance A 2.669/2.663, B 2.688/2.704.
      p99 TTFT A 800/739, B 926/848 ms; p99 TPOT A 30.3/29.8, B 27.8/24.2 ms (limits 1609 / 55.3).
      Server logs confirm k=3 (num_spec_tokens=3). Outputs: agents/leader-45/out/abba1/.
    - job 416, FP8-activation condition (task_definition): GSM8K-250, same job, fresh server each, k=3:
      record config 94.8% (5 no-answer) vs bf16-activation reference of the same code (T29_W8A8=0 T31_E4M3=0)
      93.2% (6 no-answer): +1.6 pt, within the 2.0-pt rule. 128K long request on the record config: pass (200,
      45.2 s). Outputs: agents/leader-45/out/fp8ab/.
  conclusion: Interpretation: the gain (+3.8%) is above the microbench estimate (+1.5-2.5%); it is ~1.0 ms/step of
              the ~26 ms decode step, i.e. ~transfer 0.7x of the chain's 1.38 ms. Why it was missed: gate_up kernels
              were compared at M=48 (k=2); T37's k=3 moved decode to M=64 onto Marlin's step. Lesson for the
              run: re-check GEMM dispatch whenever the decode width changes.
validity:     Reviewed against TASK.definition:
    - FP8 activations: gate_up joins the checkpoint-scheme W8A8 (dynamic per-token-group e4m3, group 128,
      128x128 block weights, fp32 cross-block accumulation) at 48<M<=96. That widens the FP8 M range, so it
      needs and has its own same-job GSM8K-250 A/B (job 416, +1.6 pt vs bf16 reference).
    - No change to communication, residual stream, norms, attention/GDN, KV, lm_head or speculation
      (verification exact).
    - Generic: the threshold is Marlin's kernel step and applies at any load. Nothing keyed to the benchmark.
    - Boundary gate: GDN state/page code untouched, so T37-002's boundary fix is carried as is; not re-run.
      128K gate re-run: pass.

official #1 (2026-11-11T10:32): 1173.4 tok/s at c=16 (mean of 2 trials: 1174.7 / 1172.1, CV 0.2%) (+84.5% vs BASE), GSM8K 94.8%, MTP acceptance 2.678, TTFT p50 149 ms, TPOT p50 12.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
