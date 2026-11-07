# Task definition (run test-3). Refines task/TASK.md; where they differ, this file wins.

## Objective and gates
- Score: output tok/s at concurrency 16 on the official benchmark (official_eval only). Noise:
  differences < 3% are not results.
- Latency gates: TTFT p99 and TPOT p99 must not be more than 20% worse than **this run's BASE
  official numbers** (as computed by official_eval), not TASK.md's figures.
- Accuracy gate: GSM8K-250 >= 89.6%.

## Correctness
- Same function as the baseline: floating-point reordering and different kernels are OK; no
  approximation (no lossy quantization, skipped layers, truncated vocab, early exits).
- Weights stay FP8 and the KV cache stays FP8: nothing lower precision, no higher-precision copies
  that change the memory picture unfairly.
- **Activations stay as in the baseline (bf16 compute through the weight-only FP8 path). FP8
  activations / W8A8 / true FP8 compute instead of bf16 activations are FORBIDDEN.**
- Communication stays at the baseline's precision (e.g. no FP8/quantized all-reduce): it is an
  approximation.
- Speculative decoding: method and draft count may change only if exact (verification against
  the target model; greedy output identical to the target's). Approximate acceptance (typical
  acceptance, relaxed thresholds, etc.) is forbidden.
- Sampling as the benchmark requests (temperature 0, max tokens, chat template, thinking as
  requested). Outputs never cut short or altered; every request answered in full.
- The server stays a general OpenAI-compatible server; the reasoning parser (qwen3) and tool
  parser (qwen3_coder) stay enabled and working.

## Hard constraints
- Fixed: model, benchmark, dataset, harness, accuracy gate, hardware (these 2 GPUs). Agents never
  modify task/.
- --max-num-seqs >= 16.
- Max context may shrink and the KV budget may change only if one 128K-token request still fits
  and is served correctly (a leader claiming this must show it was tested).
- Parallelism (TP/PP/hybrid) may change. Other serving flags (via serve_args.txt) and engine code
  are open.
- Any change that affects memory use (KV budget, max context, max-num-seqs / batch sizes, CUDA
  graph sizes, spec-decode draft count, parallelism, ...) must be shown not to crash or OOM under
  the benchmark load and with one 128K request. Pure kernel or scheduling changes don't need the
  128K test.

## Loopholes (forbidden)
- Caching or precomputing outputs (or prefill results) for the benchmark prompts; prefix caching
  tuned to the benchmark (generic default prefix caching is fine).
- Shortening, truncating or altering outputs; changing temperature, or thinking on/off.
- Inexact speculative decoding.
- No behavior keyed to the benchmark's concurrency, prompts or seed (no detecting the benchmark).
  c=16 is the target load and tuning for it is fine, but other loads must still work correctly.
- Disabling the reasoning or tool parsers; max-num-seqs below 16.
- Anything tuned to seed 0 or these 400 prompts (e.g. a draft model / n-gram table built from
  ShareGPT benchmark data).
When unsure whether something is allowed: ask the manager before building on it.

## Rulings (human, round 1)
1. FP8 draft lm_head (`draft_lm_head_quantization="fp8"`): ALLOWED, only inside the MTP drafter.
   Verification stays exact against the target at baseline precision (identical output). The
   target model's activations stay as in the baseline.
2. Reduced-vocab draft lm_head: ALLOWED if the token frequencies/ids come from non-benchmark text
   (never ShareGPT or any benchmark data) and verification stays full-vocab and exact.
3. Env-var or config defaults changed in code (in change.diff): ALLOWED if they are generic
   defaults, not keyed to the benchmark (its prompts, seed or concurrency).
