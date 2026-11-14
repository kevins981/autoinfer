# Task definition (run framework-eval). Refines task/TASK.md; where they differ, this file wins.

## Objective and gates
- Score: output tok/s at concurrency 16 on the official benchmark (official_eval only). Noise:
  as in task/TASK.md: no minimum gain; a small gain counts if same-job, interleaved, repeated
  A/Bs (each on a fresh server) show it consistently.
- Latency gates: TTFT p99 and TPOT p99 must not be more than 20% worse than **this run's BASE
  official numbers** (as computed by official_eval), not TASK.md's figures.
- Accuracy gate: GSM8K-250 >= 89.6%.

## Correctness
- Same function as the baseline: floating-point reordering and different kernels are OK; no
  approximation (no lossy quantization, skipped layers, truncated vocab, early exits). One
  exception: the checkpoint's own W8A8 activation scheme (below).
- Weights stay FP8 and the KV cache stays FP8: nothing lower precision, no higher-precision copies
  that change the memory picture unfairly.
- **Activations (changed by the human, 2026-11-09):** bf16 activations through the weight-only FP8
  path (the baseline) stay allowed. **In addition, the checkpoint's own FP8 scheme is ALLOWED:
  W8A8 with dynamic per-token-group FP8 (e4m3) activation quantization, group size 128, against
  the checkpoint's 128x128 block-scaled FP8 weights** (`quantization_config`: activation_scheme
  dynamic, weight_block_size [128, 128]); this is what the model was published for and how it
  runs on GPUs with block-scaled FP8 kernels. It may be used per layer or per batch size (e.g.
  FP8 compute only for prefill/mixed steps). Conditions:
  - Nothing coarser or lower: no per-tensor activation scales, no larger groups, no static
    (calibrated) activation scales, no FP4/INT4 or other lower formats.
  - Accumulation within each 128-element K-block may use the FP8 tensor cores' own accumulator;
    partial sums are scaled and accumulated in fp32 across K-blocks.
  - Accuracy: besides the usual GSM8K-250 gate, a record using FP8 activations runs the same
    GSM8K-250 subset (`gsm8k.py`, default n) in the same job as a bf16-activation reference of the
    same code, and must be within 2.0 points of it. (Changed by the human, 2026-11-10: was the full
    1319-question set. The sharing rules and addenda below still apply; read "GSM8K-1319 A/B" there
    as "the FP8-activation GSM8K A/B". A/Bs already run at n=1319 remain valid.)
  - Everything else stays: residual stream, norms, attention/GDN math, KV, communication and the
    target's lm_head are unchanged by this ruling (communication stays at baseline precision).
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

## Ruling (human, 2026-11-08, for v2 runs): trained draft heads
Retraining or distilling the MTP draft head (same role: it only proposes tokens; verification
stays exact, so outputs are unchanged) is allowed. Its training or tuning data must not be the
benchmark's data: not ShareGPT, not the benchmark's prompts, not anything selected because it
resembles them. It must generalize to other workloads. The validity review and the manager check
where any trained or tuned artifact's data came from.


## Ruling (human, 2026-11-08): nothing may be fitted to the benchmark
No change may be trained, tuned, calibrated or selected on the benchmark's data: not ShareGPT,
not the benchmark's prompts, not anything chosen because it resembles them. This covers learned
weights, tuned thresholds and heuristics, lookup tables, and picking between variants. Any such
data must be generic, and the result must generalize to other workloads. The benchmark measures;
it is never a training or tuning set.

## Further rulings
Questions this file doesn't settle go to the manager, who rules from the principles above (exact
math up to floating-point reordering; precision; exact verification for anything draft-side;
nothing fitted to the benchmark) or asks the human, and adds the ruling here.
