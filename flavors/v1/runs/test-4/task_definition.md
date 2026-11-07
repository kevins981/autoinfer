# Task definition (run test-4). Refines task/TASK.md; where they differ, this file wins.
# Agreed with the human (via proxy qwen38-opt-12), 2026-11-07.

## Objective and gates
- Score: output tok/s at concurrency 16 on the official benchmark (official_eval only; 2 trials, mean).
- No minimum gain: a change beats the reference if its gain shows up consistently (per-trial
  official numbers; same-job A/B vs the reference, repeated when close; see TASK.md's noise
  notes), not if it clears a % cutoff.
- Latency gates: TTFT p99 <= 1609 ms and TPOT p99 <= 55.3 ms (BASE official 1341 / 46.1 ms, +20%).
- Accuracy gate: GSM8K-250 >= 89.6% (BASE 94.4%).
- Long-request gate: one 128,000-token request returns generated tokens; the server still answers afterwards.

## Correctness
- Same function as the baseline: floating-point reordering and different kernels are OK. No
  approximation: no lossy quantization, skipped layers, truncated vocab or early exits.
- FP8 weights and FP8 KV cache: nothing lower precision, and no higher-precision copies that
  change the memory picture unfairly.
- Activations stay bf16, as in the baseline (weight-only FP8 via Marlin, bf16 compute).
  FP8 activations, W8A8 and true FP8 compute are FORBIDDEN in the target model. (The drafter is
  exempt, see below: its output only affects acceptance, never the served tokens.)
- Communication (all-reduce, all-gather, ...) stays at the baseline's precision: no FP8 or
  quantized comms. Bitwise-lossless compression is allowed.
- Speculative decoding must stay exact: verification against the target model, full-vocab, so
  the output is what the target would produce. Approximate acceptance (typical acceptance,
  relaxed thresholds) is forbidden. The drafter itself (method, number of draft tokens, its
  internals) may change. Specifically allowed: an FP8 draft lm_head (inside the drafter only), and
  a reduced-vocab draft head if its token ids come from non-benchmark text.
  Ruling r1 (manager, for leader-2): vLLM's draft_lm_head_quantization="fp8" (W8A8, per-token
  activation scales, FP8 compute) is allowed: it is the FP8 draft lm_head the human allowed,
  inside the drafter only.
- Sampling as requested (temperature 0, max tokens, chat template, thinking as requested).
  Outputs are never cut short or altered; every request is answered in full.
- The server stays a general OpenAI-compatible server; the qwen3 reasoning parser and the
  qwen3_coder tool parser stay enabled and working.
- How correctness is judged: by the math (exact, or a bounded-error reordering), numeric checks
  of the changed op against the old op on the same inputs, GSM8K, and MTP acceptance. Identical
  output text is NOT the test, because batch composition flips near-ties. Token-for-token
  comparison applies only to bitwise-exact changes run one request at a time.

## Hard constraints
- Fixed: model, benchmark, dataset, harness, gates, hardware (these 2 GPUs). Agents never modify task/.
- --max-num-seqs >= 16.
- Max context may shrink and the KV budget may change only if one 128K-token request still fits
  and is served correctly (tested).
- Any change that affects memory use (KV budget, max context, batch sizes, CUDA graph sizes,
  draft count, parallelism, buffers) must be shown not to crash or OOM under the benchmark load
  and with one 128K request. Pure kernel or scheduling changes don't need the 128K test.
- Parallelism (TP/PP/hybrid), other serving flags (via serve_args.txt) and engine code are open.

## Loopholes (forbidden)
- Caching or precomputing outputs or prefill results for the benchmark prompts; prefix caching
  tuned to the benchmark. Generic prefix caching is fine.
- Shortening, truncating or altering outputs; changing temperature or turning thinking on/off.
- Inexact speculative decoding.
- Behavior keyed to the benchmark (its prompts, its seed, detecting it). c=16 is the target load
  and tuning for it is fine, but other loads must still work correctly.
- Disabling the reasoning or tool parsers; max-num-seqs below 16.
- Anything tuned to seed 0 or these 400 prompts (e.g. a draft model, vocab or n-gram table built
  from ShareGPT benchmark data).
- Env/config defaults changed in code are allowed only if they are generic, not keyed to the benchmark.
When unsure whether something is allowed: ask the manager before building on it.
