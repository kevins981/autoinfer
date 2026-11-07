# Unmeasured ideas: qwen38-27b-fp8-2x4090

Ideas for this task with **no measured result of the change itself**: untested, or ruled out only
by an estimate. Not part of the answer key and **not used for grading** (a run that skips them
missed nothing we know of). A component measurement of the bottleneck (e.g. how much time the
norms take) is not a measurement of the change. When a run measures one of these, it moves into
`answer_key.md` as a node, with its result. Same format as the key's nodes; `builds_on` may name
key nodes or other ideas here.

### `l2-prefetch`: prefetch the next GEMM's weights into L2 during the all-reduce
- builds_on: host-ar
- status: open
- results: on host-ar: est. +3-5%, low confidence (`v1/test-3` ideas R2-C, R3-D). `component`: L2-hot
  Marlin at M=48 is 41-48 us vs ~50-56 us in-server, which caps it near +3%
  (`v1/test-4:brainstorm/r1/adjacent_field.md:90`).
- note: do it after `gdn-replay` (GDN state writes spill through L2).

### `ar-compression`: bitwise-lossless compression of the all-reduce payload
- builds_on: host-ar
- status: open
- results: on host-ar: est. +2-3%, depends on compressibility (`v1/test-3` idea R3-C).

### `small-spec-state`: one GDN spec block per request
- builds_on: gdn-replay
- status: open
- what: with replay, only the log slot is needed per request, so the per-request spec-state blocks
  (F5) could shrink from k+1 to 1.
- results: on gdn-replay: untested; frees spec pages so `draft-depth-3` fits 16 requests in 5 GiB
  (`v1/test-3` ideas R3-B v2, KB17). Risk: silent state corruption from an unknown reader.

### `draft-depth-3-small-state`: 3 draft tokens once spec state is small
- builds_on: small-spec-state + draft-depth-3
- status: open
- results: est. +2-5% (`v1/test-3` idea R3-A). Acceptance rises 2.35 → ~2.75 (`draft-depth-3`); the
  decode step costs +12.6% for +16.7% tokens on test-4's champion (`v1/test-4:KB13`).

### `pp2`: pipeline parallel 2 instead of TP=2
- builds_on: BASE
- status: ruled out by estimate
- results: est. worse: doubles weight streaming per token (`v1/test-4:KB10`).

### `uneven-tp`: uneven TP split to offset GPU 0's slower clock
- builds_on: BASE
- status: ruled out by estimate
- results: est. ~0-1.5% (`v1/test-3:KB8`, brainstorm r2).

### `spec-glue-graph`: capture the spec-decode glue in one CUDA graph
- builds_on: BASE
- status: open
- what: drafter passes, sampling and attention-metadata builds run eagerly between graphs.
- results: on host-ar + draft-head-diet + sharded-sampling: ~1.6 ms/step of real CPU idle, est.
  +3-4% (`v1/test-4:KB12`). Sub-options, all estimates: skip no-op eager launches in decode
  (`_zero_kv_blocks`, `_apply_write` x5, mamba pre/post), 0.5-1.0 ms
  (`v1/test-4:agents/leader-2.w2/REPORT_job1.md:32`); pure launch fusion bounded at 0.1-0.2 ms
  (`v1/test-4:T2-007`); in-place FlashInfer draft-metadata update, ~1 ms (`v1/test-3` idea R1-H).
- caution: part of the step-start gap in node-level nsys is an artifact (F8).

### `combine-frontiers`: triton-attn (+ draft-vocab-subset) on test-3's frontier
- builds_on: triton-attn + prefill-graphs + gdn-replay
- status: open
- results: the two tied frontier stacks share only host-ar and draft-head-diet; their other parts
  attack different costs (attention planning vs prefill graphs / GDN state), est. +5-7% if additive
  (`evals/baseline/run-1/judge-a.md`).

### `ba-proj-splitk`: better GEMM choice for GDN `in_proj_ba` at tiny M
- builds_on: BASE
- status: open
- results: `component`: the bf16 GEMM takes 26 us vs 5 us at M=2-8 (cuBLAS picks a no-split-K kernel),
  only on tail steps; ~3.4 ms on those steps (`baseline/run-1:tsv:22`). Small expected gain.

### `decode-ar-overlap`: hide decode all-reduces behind compute
- builds_on: host-ar
- status: open
- results: untested (GEMM-epilogue streaming of the AR, or two-microbatch decode)
  (`baseline/run-1:final.md`). Marlin's persistent grid blocks concurrent kernels (F9).
