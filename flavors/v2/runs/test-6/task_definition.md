# Task definition (run test-5, resumed from test-3). Refines task/TASK.md; where they differ, this file wins.

## Objective and gates
- Score: output tok/s at concurrency 16 on the official benchmark (official_eval only). Noise:
  as in task/TASK.md: no minimum gain; a small gain counts if same-job, interleaved, repeated
  A/Bs (each on a fresh server) show it consistently. (The "< 3% are not results" rule of
  test-3 was removed by the human at the resume, 2026-11-08.)
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

### Clarification (manager, 2026-11-08, under the trained-draft-head ruling)
Prompts generated by the target model itself (Magpie-style free sampling from the chat template,
or self-instruct from a generic, hand-written topic list) with the target's own replies as labels
are acceptable training data. Conditions: no external chat dataset, no benchmark prompts or
ShareGPT, no filtering/selection or topic list chosen to resemble the benchmark, and model
selection on a held-out split of that self-generated data only (benchmark acceptance is measured
only for evaluation). The record must document the data pipeline (generator script, topic list,
counts). The human may override.

### Ruling (manager, 2026-11-09): moving RMSNorm's per-row 1/rms past the following linear
ALLOWED as floating-point reordering: y = inv_rms[row] * (W · bf16(h ⊙ g)) applied in the fp32
GEMM epilogue, instead of W · bf16(bf16(h*inv_rms) * g). Conditions: inv_rms computed in fp32
from the same bf16 h as the baseline; the GEMM input stays bf16 (never FP8 or lower) and
accumulation stays fp32; the residual stream is unchanged; the record reports max-abs/relative
diff vs the stock norm+GEMM on real activations, and GSM8K. Same logic applies to other
per-row/per-column scale fusions that only move a rounding point. The human may override.

### Ruling (manager, 2026-11-09): FP8 MTP fc inside the drafter
ALLOWED, same logic as ruling 1: the MTP fc (used only by the drafter) may run in FP8 (block-scaled like the
checkpoint, or per-output-channel scales as the FP8 draft lm_head; amended 2026-11-09), since it only proposes tokens and verification stays exact against the target.
Preferably it replaces the bf16 fc (memory falls); if a bf16 copy is kept as well, it is a memory
change (no OOM under load + 128K test). The target model's weights and activations are unchanged.
The record reports MTP acceptance before/after. The human may override.

### Ruling (manager, 2026-11-09): screened (low-rank) draft lm_head inside the drafter
ALLOWED under rulings 1-2 and the trained-draft-head ruling + clarification: a draft-only candidate screen (e.g. a PCA
projection of drafter hidden states selecting top-k candidates, then exact rescoring with the draft lm_head) only changes
which token is proposed; verification stays exact and full-vocab against the target. Conditions: the projection is fitted
only on target-self-generated data per the clarification (no benchmark prompts/ShareGPT, no benchmark-resembling
selection), model selection on a held-out split of that data; the record documents the data pipeline (script, counts,
split) and reports MTP acceptance before/after; memory added -> no OOM under load + 128K test. The human may override.

## Ruling (human, 2026-11-09, relayed by the designer): fine-grained prefix caching for the hybrid model
APPROVED: generic fine-grained prefix caching for the hybrid (GDN) model is allowed: `--prefix-match-unit`,
`--enable-mamba-shared-prefix-checkpoint`, and a prefix-state cache built from live traffic (T21-1). Condition: nothing
may be sized or tuned to this particular 45-token system prompt or to the benchmark's prompts: no match units,
checkpoint positions or warm-ups chosen for them. The mechanism must work for any shared prefix.
(Manager's check for validity reviews: the record shows the mechanism working on at least one different shared prefix,
e.g. another system prompt or a different length, and cites where match units / checkpoint positions come from.)

## Manager clarification (2026-11-09): sharing the GSM8K-1319 FP8-activation A/B
The FP8-activation condition (Correctness) measures the accuracy cost of the FP8 activation numerics. One same-job
GSM8K-1319 A/B (FP8 vs bf16-activation reference of the same code) may serve several records whose FP8 activation
code is identical: same quant kernel, same GEMM kernels and configs at every M they run, same layers. Changes that are
exact by this definition may differ between the records: speculation depth with verification unchanged, bitwise-equal
state handling, scheduling. Each record cites the shared A/B job and lists its differences from the code that was
tested. Its own official eval still runs the GSM8K-250 gate. A change to the FP8 code (a different GEMM kernel, a
quant fusion, a different set of layers or M range) needs its own A/B.
Addendum (manager, 2026-11-09, asked by T28/T29): "identical FP8 activation code" also covers (a) a fused producer
(e.g. norm -> quant, silu-mul -> quant) whose outputs are shown BITWISE equal to the unfused path (the producer's bf16
output where it is still consumed, the fp8 bytes and the fp32 group scales) on real activations captured from a served
run, covering every layer type and the M range where it runs; and (b) GEMM configs that differ only in tiling or split-K
on K-block boundaries (fp32-level reordering of the per-block promoted sums, as for any GEMM). The record cites the
bitwise test. Anything that changes the quant math, the group size, the layers or the M range still needs its own A/B.
Addendum 2 (manager, 2026-11-10, asked by T31): a record whose FP8 code is identical to a tested record's but used on a
strict SUBSET of its layers or M range, with the rest falling back to the champion's already-validated path (bitwise
Marlin bf16-activation or an FP8 path that has its own A/B), may reuse the tested record's GSM8K-1319 A/B. It cites the
A/B and states the subset. Widening the range or adding layers still needs a new A/B.

## Clarification (manager, 2026-11-10, asked by T36): load-adaptive draft depth
A draft depth (or verify budget) chosen from the live engine state (number of running requests, drafter confidence, the measured cost curve) is a generic policy and is allowed, provided verification stays exact (same outputs as k=2 up to fp reordering) and nothing in it detects or is keyed to the benchmark: no thresholds or constants tuned to concurrency 16, these 400 prompts or seed 0. The policy must also be sane at other loads (it is judged by a record running at c=16 only, but it must be explainable without the benchmark). The human may override.

## Ruling (manager, 2026-11-11, asked by T42): certified screening of the TARGET lm_head
ALLOWED, provided the greedy decision is bitwise the baseline's: a screening plane (e.g. b-bit integer copy of the lm_head with a rigorous error bound) may only select a candidate set; the candidates' logits are then computed exactly at the baseline's precision (the bf16 lm_head, or an exact copy of it), and the argmax is taken over them with the baseline's tie-breaking. A row where the certificate fails falls back to the full exact lm_head. The reference argmax must be provably inside the candidate set (a bound, not an empirical margin). Memory: the screening plane counts against the same GPU budget (a +0.6 GiB startup increase has been seen to OOM, KB111); a host-backed rescoring variant must not add decode latency. The human may override.
