# Task definition (run review-eval) — DRAFT, pending the human's agreement

Base: task/TASK.md (objective, metric, noise, gates, fixed/open lists, correctness). Where this file
differs, this file wins. Status: draft 2026-11-15, open questions marked [Q].

## Objective
Max output tok/s at concurrency 16 on the official harness (2 fresh starts x 400 ShareGPT prompts,
mean), Qwen3.8-27B-FP8 on 2x RTX 4090. Champion to beat: BASE 635.9 tok/s.

## Correctness
- Same function as the baseline up to floating-point reordering; no approximation (no lossy
  quantization, skipped layers, truncated target vocab, early exit).
- FP8 weights, FP8 KV cache, bf16 activations — or the checkpoint's own W8A8 scheme (dynamic
  per-token-group-128 e4m3 activations vs 128x128 block weights, fp32 accumulation across K-blocks),
  per layer / per batch size allowed. Comms, norms, attention/GDN, KV cache, target lm_head as baseline.
- Speculative decoding must stay exact (target-model verification, greedy-equivalent output).
- Evidence of correctness: by construction / numeric op checks (bitwise or max-abs-diff), MTP
  acceptance, and GSM8K-250 >= 89.6% (GATES=long,gsm8k) for any change that can alter the math.
  Token-for-token equality only meaningful for bitwise-exact changes run one request at a time.
- Sampling as requested: temperature 0, max_tokens, chat template, thinking per request; outputs
  never cut short or altered. Sanity check: total output tokens of a bench stay close to BASE's.

## Hard constraints
- Model, weights, architecture fixed; tokenizer and chat template unchanged.
- Hardware: these 2 GPUs only; no clock/power changes (needs root anyway).
- Harness, dataset, gates frozen; nobody modifies task/.
- Prefix caching stays on (human ruling 2026-11-15).
- Server stays a general OpenAI-compatible server: tool-call and reasoning parsers keep working.
- Max context stays >= 131072 (the 128K gate must pass).
- Latency guard: TTFT p99 and TPOT p99 within +20% of BASE (1341 ms / 46.1 ms).
- Startup < 15 min per server start.

## Open
Any engine code, build, kernels (AOT or JIT), parallel scheme (TP/PP/other), serve flags via
serve_args.txt (batching limits, chunked prefill, KV budget, CUDA graphs, backends, env vars),
speculative method and draft count — within the above.

## Loopholes (forbidden unless noted)
1. Anything specific to the benchmark: caching/precomputing outputs or prefix-cache warming with its
   prompts, detecting it, tuning to seed 0 or these 400 prompts. Tuning to the workload's *shape*
   (concurrency 16, ~260 in / ~228 out tokens) and to the model's GEMM shapes is allowed.
2. Anything trained, tuned or calibrated (e.g. a draft model, ngram tables) must not use the
   benchmark's data (ShareGPT split used by the harness, GSM8K questions).
3. Changing outputs' length (biasing EOS, lowering max_tokens) — forbidden.
4. Disabling thinking/parsers or the chat template to cut tokens — forbidden.
5. Turning off prefix caching — forbidden (ruled).
6. Cherry-picking harness runs — see the measurement rule below. [Q3]
7. Drafter-side approximations (lower-precision draft head, truncated draft vocab, a different/extra
   draft model): output stays exact by verification. [Q1, Q2]
8. Lowering KV budget / max context to grab memory that breaks the 128K gate — the gate decides.

## Measurement / promotion rule [Q3, proposed]
- Every harness run of a record is filed in bank/<id>/eval/; none may be dropped.
- A record is promoted when its harness score (mean of its 2 starts) beats the champion's AND both
  starts exceed the champion's mean, checks pass (latency, 128K, GSM8K if math can change), validity
  reviewed, mechanism shown. If the record has several harness evals, its score is the mean of all.
- For a claimed gain < 1%, a second harness eval is required (all 4 starts above the champion's mean).
