# Unmeasured ideas: qwen38-27b-fp8-2x4090

Ideas for this task with **no measured result of the change itself**: untested, or ruled out only
by an estimate. Not part of the answer key and **not used for grading** (a run that skips them
missed nothing we know of). A component measurement of the bottleneck (e.g. how much time the
norms take) is not a measurement of the change. When a run measures one of these, it moves into
`answer_key.md` as a node, with its result. Same format as the key's nodes; `builds_on` may name
key nodes or other ideas here.

### `ar-norm-fusion`: fuse residual-add + RMSNorm into the host all-reduce kernel
- builds_on: host-ar
- status: open
- results: on host-ar: untested; ~1.6 ms/step of norms etc. (`v1/test-4` reports/final.md).

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

### `reduced-vocab-draft-head`: draft head over a reduced vocabulary
- builds_on: draft-head-diet
- status: ruled out by estimate
- results: on draft-head-diet: est. net ≈ 0 once the draft head is FP8 (`v1/test-3` idea R1-J,
  ideas.md:43). A measurement overrides this.

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

### `dequant-cublas-prefill`: dequantize + cuBLAS for large-M (prefill) GEMMs
- builds_on: BASE
- status: ruled out by estimate
- results: est. ≤ 2% overall (`v1/test-3:brainstorm/r2/adjacent_field.md:75`).

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

### `marlin-tuning`: in-kernel Marlin GEMM efficiency
- builds_on: BASE
- status: open
- results: no change tried. `component`: near its floor: the M=48 penalty vs M=3 is ~1.6 ms/step
  (`v1/test-3:KB14`); L2-hot Marlin is close to its own floor
  (`v1/test-4:brainstorm/r1/adjacent_field.md:90`). Small upside (F7).

### `ce-ar-prefill`: copy-engine full-duplex all-reduce for large (prefill) tensors
- builds_on: BASE
- status: open
- results: est. +1.5-3%, but half the prefill AR time is GPU 0 skew (F2) (`v1/test-3` idea R2-E).
  Complements `host-ar` (small sizes).

### `flashinfer-pcie-ar`: `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`
- builds_on: BASE
- alternative_to: host-ar
- status: open
- results: a 5-minute check, never run (`v1/test-3:brainstorm/r1/systematic_engineer.md:99`).
