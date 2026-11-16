# Task definition (run review-eval; agreed for framework-eval, reused with the human's later rulings, 2026-11-15). Refines task/TASK.md; where they differ, this file wins.

## Objective and gates
- Score: output tok/s at concurrency 16 on the official benchmark (the task's harness, run by the
  teams: 2 fresh starts x 400 prompts + the 128K request). Noise: as in task/TASK.md: no minimum
  gain; a small gain counts if it shows up consistently across fresh starts.
- Latency gates: TTFT p99 and TPOT p99 must not be more than 20% worse than BASE's numbers
  (task/baseline/official/, the harness's own result for BASE).
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
  - Accuracy: the usual GSM8K-250 gate. (Changed by the human, 2026-11-15: the separate
    same-job GSM8K comparison against a bf16-activation reference is dropped.)
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

## Ruling (human, 2026-11-15): prefix caching stays on
Prefix caching stays on (the default). Turning it off (`--no-enable-prefix-caching`) is not allowed
(also in task/TASK.md). An earlier manager ruling that allowed it is overruled.

## Ruling (human, 2026-11-16): when a record beats the champion
A record's score is the mean of all its starts (the harness: 2 fresh starts x 400 prompts per run). It beats the champion when that mean is higher **and every start is above the champion's mean**. A gain under 1% needs one more start, in a new GPU job (`TRIALS=1`), with all 3 starts above the champion's mean. (Jobs drift up to ~1.6%, so one lucky job can fake a ~1% gain.)

## Ruling (manager, 2026-11-16): scale floors in W8A8 activation quantization
A minimum on the per-group activation scale (e.g. vLLM's silu_and_mul_per_block_quant floor,
scale >= 1/(448*512) = 4.36e-6, vs the unfused path's amax >= 1e-10) is an implementation detail of
the allowed dynamic per-token-group-128 scheme, not a coarser scheme: it only changes groups whose
values are all below ~1.95e-3, and within them only values below e4m3's normal range at that scale
(~7e-8) lose relative precision. Allowed, if disclosed in the record's validity; the GSM8K gate applies.
(Raised by T2-006, reviews/T2-5.)

## Ruling (manager, 2026-11-16): what is measured must be what is filed
A champion candidate's harness starts must run the code in its change.diff. If the code changes
after the measurement (even guards), measure the filed code before promotion: one start in a new
GPU job (TRIALS=1) above the champion's mean suffices when the change is not meant to alter the
measured path; the record's score is then the mean of all its starts.

## Ruling (manager, 2026-11-16): no fp16 accumulation in W8A8 GEMMs
"The FP8 tensor cores' own accumulator" (W8A8 ruling) means the fp32-accumulate form of the FP8 MMA
(whatever internal precision the hardware gives it). Choosing the fp16-accumulate MMA variant
(e.g. sm_89 `mma ... .f16` accumulator) is NOT allowed, for any M: it is a deliberate lower-precision
accumulator (10-bit mantissa, overflow above 65504; e4m3 products reach ~2e5), not floating-point
reordering, and falls under "nothing coarser or lower". An overflow fallback does not change this.
(Asked by T8; the human may overrule.)
