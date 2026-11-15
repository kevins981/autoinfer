team:         T3
builds_on:    T1-019
performance:  "unofficial: PASS +2.11% vs T1-019 (863.0/874.5 vs 850.8)"; worker A/B job 186 +0.9% (noise)
summary:
  hypothesis: with batch-sharded sampling, every decode step all-to-alls the target's partial-vocab logits (~24 rows x 124160 x
              bf16 ~6 MB per direction): one NCCL SendRecv of ~455 us on both GPUs (T3 job 153), ~2% of the decode step. For
              all-greedy batches only (max, argmax) per row is needed: argmax commutes with the vocab split (combine = max with
              lowest-index tie-break), like the drafter's local argmax reduction (T1-001).
  change:     worker t3-w1, workspace t3-w3 (on T1-019): model_runner.sample() takes a local-argmax path when the whole batch is
              greedy and unprocessed (temperature 0, no logprobs, no logits processing incl. penalties/bias/bad words/min_tokens/
              thinking budget, no grammar, plain rejection sampler); lm_head local on all rows, per-shard argmax, all-gather of
              (value, index), identical greedy verify on every rank (new Triton _greedy_verify_kernel mirroring _rejection_kernel +
              _insert_resampled_kernel). Otherwise the old path. Env VLLM_TARGET_LOCAL_ARGMAX (default 1).
  results:    Observed: offline (job 181) 0 mismatches vs torch.argmax over 7573 full 248K rows with forced ties; verify kernel vs a
              Python reference: 0 bad in 300 batches. Server c=1 greedy GSM8K-16 (job 186, 4686 tokens), OFF/ON/ON/OFF: 16/16 token
              ids identical in all pairs. nsys (job 188): SendRecv 0 (was ~1/step, 455 us), logits-stats and rejection kernels gone,
              greedy verify 1.5 us; decode period 22.06 ms.
  conclusion: Interpretation: the removed logits all-to-all (~0.45 ms per decode step plus stats/rejection kernels) carries over end to end (+2.1%, both starts above HEAD); p99 TTFT 900 ms, TPOT 37.9 ms.
validity:     Exact: same argmax as the full-row path (ties to lowest index, matching torch.argmax), verified bitwise; outputs
              identical at c=1. Path keyed only on request sampling params (generic), falls back otherwise; branch decided on host
              state identical across ranks. Known difference: NaN logits (old path maps NaN to -inf); NaN logits do not occur in
              valid serving. No memory change. Reviewed by leader-3.

unofficial #1 (2026-11-15T13:04): PASS: +2.11% vs HEAD T1-019 (B 863.0 / 874.5 vs reference mean 850.8 over 2 starts, 0 run in this job; 200 prompts); p99 TTFT 900 ms / TPOT 37.9 ms, MTP acceptance 2.338 (unofficial/1/unofficial.json)

official #1 (2026-11-15T14:08): 908.5 tok/s at c=16 (mean of 2 trials: 906.2 / 910.8, CV 0.4%) (+42.9% vs BASE), GSM8K 93.2%, MTP acceptance 2.319, TTFT p50 174 ms, TPOT p50 16.2 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
