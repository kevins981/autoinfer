team:         T1
builds_on:    T1-014
performance:  "official #1: 837.0 tok/s (-0.7% vs T1-014 842.7, separate jobs; between-job variation unmeasured): no evidence of a win. unofficial: mixed, -0.39% (ABBA 200 prompts vs T1-014, job 62: B 817.9/804.2 vs A 809.2/819.2); GSM8K-50 on B 90.0%"
summary:
  hypothesis: Row-split (T1-013) only covered 256-511-token steps. Steps of 512-2048 tokens run eager, ~12% of run time,
              and are GPU-bound. They should gain from the same AR/GEMM overlap. On the side stream, NCCL beats the
              spinning HOSTMEM AR, whose blocks compete with the overlapped GEMM for SMs.
  change:     T1-014 + agents/leader-1.w1/rs2/delta-vs-T1-013.diff: VLLM_ROWSPLIT_MAX_TOKENS (default 2048 = the
              scheduler's max), VLLM_ROWSPLIT_NCCL=1 (side-stream ARs forced to NCCL via hostmem FORCE_NCCL flag).
  results:    Observed (worker job 61, 1 start/arm, 150 prompts, steplog): median step 512-799 tok 163.8 -> 154.8 ms,
              800-1199 tok 209.4 -> 172.6 ms, 1600+ 418.7 -> 376.1 ms; buckets <512 unchanged; 761.0 -> 781.0 tok/s.
              ABBA job 62 (agents/leader-1/jobs/c7): mixed -0.39%; acceptance B 2.352/2.331 vs A 2.340/2.358; TTFT/TPOT p99
              B 908/900 / 37.0/35.6 vs A 1021/907 / 38.1/36.1 ms; GSM8K-50 on B 90.0%. EngineDeadError in 2_B/server.log is
              the harness's forced shutdown after the bench (logged after 'shutdown triggered'), not a crash.
  conclusion: Interpretation: napkin gain ~+1% (>=512-token steps ~12% of time x ~10% faster) is below the ABBA's
              between-start noise (~1%) at 200 prompts, so the ABBA can neither show nor rule it out. Requesting the
              official eval (CV ~0.2% across its trials) as the more precise test; promote only if it beats 842.7.
              After official: 837.0 vs 842.7 (-0.7%, trials 838.7/835.4 vs 841.6/843.8): the per-step savings on >=512-token
              steps (job 61) don't show up end to end. Possibly the forced-NCCL side-stream AR or the eager split adds cost
              elsewhere, or job 61's +2.6% (1 start/arm) was noise. Not kept.
validity:     Python-only. Same ops, same rows, different launch order/stream: row-independent sublayers on row halves,
              bf16 NCCL all-reduce (baseline precision). Exact up to fp reordering (GEMM tiling per M may differ).
              Generic gate on token count, no benchmark-specific keys.

official #1 (2026-11-14T20:58): 837.0 tok/s at c=16 (mean of 2 trials: 838.7 / 835.4, CV 0.3%) (+31.6% vs BASE), GSM8K 93.6%, MTP acceptance 2.334, TTFT p50 160 ms, TPOT p50 17.6 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
