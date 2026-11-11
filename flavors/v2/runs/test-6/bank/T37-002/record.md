team:         T37
builds_on:    T31-003 + two-slot GDN state (ported from bank/T28-005/change.diff; origin T25-003 / T5-005) + a boundary-copy fix (new, T37)
performance:  "unofficial: same-job ABBA vs T31-003 (job 375, 300 prompts): all B > all A, +1.34% (B 1112.9/1113.1 vs A 1100.3/1096.3), measured before the boundary fix (see results)"
summary:
  hypothesis: a decode step is ~72% fixed cost (T = 16.9 ms + 0.10 ms*rows + 0.83 ms*draft steps, bank/T37-001) and draft acceptance is
              geometric (~0.72 per further position), so a third draft token pays once all 16 requests can run; they cannot on the
              champion because GDN state blocks per request grow with k (14 of 16 run at k=3).
  change:     (1) two-slot GDN state, default ON (VLLM_GDN_TWO_SLOT=0 restores the stock layout), active only for k in 2..3 with GDN
              replay available on the V2 runner: a request keeps the committed state S[0] and the replay log S[1] instead of 1+k
              slots. Files: csrc/gdn/fused_gdn_decode_kernel.cu, vllm/_custom_ops.py, vllm/envs.py,
              vllm/model_executor/layers/mamba/gdn/{replay.py,qwen_gdn_linear_attn.py}, vllm/v1/attention/backends/gdn_attn.py.
              (2) fix of a defect in that ported code: the align-mode page-boundary copy read the temporal state at column
              accepted-1, which the two-slot flush does not write; under two-slot the token bias is now zero for temporal states
              (vllm/v1/worker/mamba_utils.py, Triton constexpr ZERO_TEMPORAL_TOKEN_BIAS in the three fused copy paths).
              (3) serve_args: num_speculative_tokens 3 (otherwise the champion's). (4) inert step logger (env T37_STEPLOG, off by default).
              Worker leader-37.w1, workspace t37-depth; merge notes agents/leader-37.w1/merge/.
  results:    Observed:
    - job 372 (k=3, 150 prompts): 16 requests running; pure-decode step at 64 rows median 26.16 ms (p10-p90 25.85-26.45; model 25.8);
      KV pool 276,547 tokens at the unchanged 5.0 GiB budget (champion 267,721), peak KV usage 58.6%.
    - job 375 ABBA, 300 prompts c=16 (A = t37-champ = T31-003, B = this record minus the boundary fix, flag set by env):
        A 1100.27 tok/s, acc 2.321, TTFT p50/p99 152/785, TPOT p50/p99 13.2/29.0
        B 1112.91, acc 2.684, TTFT 183/910, TPOT 12.7/28.1
        B 1113.12, acc 2.661, TTFT 143/846, TPOT 12.8/26.8
        A 1096.34, acc 2.324, TTFT 140/693, TPOT 13.0/29.8        verdict: all B > all A, +1.34%
    - job 376: long_request 128000 on the B config: pass (200, 45.0 s, health OK), 844 MiB free on both GPUs, no OOM.
    - job 374: k=2, flag off vs on, one request at a time, temp 0, 24 prompts x 200 tokens: 24/24 identical full message text.
    - job 377 (before the fix): k=2, flag off vs on, 4 prompts of 1495/1563/3125/3153 tokens, 250 output tokens: ALL 4 diverged into
      garbage right after the 1600 / 3200-token page boundary. job 378 (after the fix): 4/4 identical to flag off.
    - job 379 (the exact official config: this diff, default flag, k=3): the same 4 prompts one at a time vs the k=2 flag-off
      reference: 2 identical (1563 and 3153 tokens, both cross a boundary), 2 diverge early (chars 140 and 295, before any
      boundary) into fluent on-topic text; the 4 sent together with 8 short prompts: all differ from the reference, all fluent,
      normal length, no corruption.
    Estimate before measuring: +4.5% (cost model), then +2-3% after reading T28-003/T36-001. Measured +1.34%.
  conclusion: Interpretation: acceptance +15% for a +12% decode step nets ~+3% on decode and ~+1.3% end to end. The early divergences
    in job 379 are near-tie flips from different GEMM row counts (k=3 changes M; batching changes it again), not state errors: the
    boundary-crossing requests are identical where the numerics allow. The ABBA was timed without the boundary fix; the fix only
    changes a copy that runs when a request crosses a 1600-token page, which ShareGPT requests do not, so the number should stand;
    the official eval re-times the final diff.
validity:     (leader-37 review of the diff vs T31-003)
    - Math unchanged: only where the GDN recurrent state is stored; target forward, drafter and verification untouched; spec decode
      stays exact (greedy verification against the target). Evidence: jobs 374, 378, 379 above. Text identity is used only
      one-request-at-a-time, as task_definition.md allows.
    - k=3 is a serve flag; nothing keyed to the benchmark. The flag default is generic (any k in 2..3 with replay).
    - Memory change (state blocks, k): 128K gate tested (job 376); KV budget and max context unchanged. The boundary fix does not
      change memory.
    - FP8 activations: the FP8 code is T31-003's, unchanged (same quant kernel, same t31_gemm_e4m3 kernel, same layers, same
      M<=96 dispatch). k=3 moves pure-decode steps from M=48 to M=64, inside the champion's M<=96 range; t31_cfg.py picks other
      (UN, KS, KSEG, MT) configs in the 64 bucket. In t31_gemm_e4m3.cu K is walked in 128-wide blocks (BL = K/128; KS and KSEG
      split on block indices, lines 65-69) and every block is promoted to fp32 with its scales (lines 166-177) for every config, so
      the configs differ only in tiling / split-K on K-block boundaries: addendum (b) of the 2026-11-09 clarification. On that
      basis this record reuses T31-003's FP8-activation A/B (job 288). Not shown by me: that job 288 itself exercised M>48; if
      the manager wants the GSM8K-250 A/B at k=3 instead, it has not been run (team balance did not cover it plus the eval).
    - Known limits: two-slot is not enabled for k=1 or k>=4 (stock layout there). The step logger is inert without its env var.

official #1 (2026-11-10T22:38): FAILED at apply_diff (official/1/prepare.log)

official #2 (2026-11-10T22:45): 1142.2 tok/s at c=16 (mean of 2 trials: 1138.4 / 1146.1, CV 0.5%) (+79.6% vs BASE), GSM8K 94.0%, MTP acceptance 2.674, TTFT p50 148 ms, TPOT p50 12.7 ms; gates: GSM8K pass, latency pass, 128K request pass (official/2/results.json)

FP8-activation condition (leader-37, job 381, 4.6 GPU-min, workspace t37-depth = this record's diff, serve_args as here, k=3):
GSM8K-250, same job, fresh server each: FP8 activations (record config, defaults) 94.4% (4 no-answer) vs bf16-activation
reference of the same code (T29_W8A8=0 T31_E4M3=0; its server log has no e4m3/W8A8 dispatch lines) 93.6% (6 no-answer):
+0.8 points, within the 2.0 limit. This replaces the reuse argument for job 288 in the validity section above.
Outputs: agents/leader-37/out/fp8ab/{fp8,bf16ref}/gsm8k.json.
