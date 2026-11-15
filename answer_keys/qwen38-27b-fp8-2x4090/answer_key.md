# Answer key: qwen38-27b-fp8-2x4090

A map of this task's search space, as far as we know it: **measured** changes, grouped by family, each
with what is known about it on each base, plus the facts that explain them. Untested ideas are not
nodes: they are in `ideas.md` until a run measures them. Merged from all runs of all flavors; a lower
bound that evolves. Format and grading rules: `EVAL.md`.

Metric: output tok/s at c=16 (TASK.md). BASE ≈ 631-636 official (varies ~1% between runs).
Evidence ids: `<flavor>/<run>:<record | KBn | path:line>`. Runs: `v1/test-3`, `v1/test-4`,
`baseline/run-1`, `v2/test-6` (seeded from test-5 at T1-042; T1-* records are test-5's),
`v3/framework-eval`, `v4/sweep-unofficial-eval` (`v4/sue`), `v4/sweep-simple-eval` (`v4/sse`);
`qwen38_opt` = the hand study (its BASE differed: 3 draft tokens, no FP8 KV).
Evidence strength, strongest first: `official` (fresh-server trials) > `abab` (interleaved same-job
A/B) > `ab` > `unofficial` (v4 unofficial_eval: 2 fresh starts, 200 prompts; tends to understate
gains, F16) > `component` (microbench / nsys of one part) > `estimate`. Gains are relative to the
record it was stacked on unless said otherwise.

## Frontier and reference curves
- **Best known: 1229.6 tok/s official, +94.7% vs BASE 631.4** (`v2/test-6:T58-001`). Cost: test-5's
  budget (to 942.2) + 1207 GPU-min in test-6 (942.2 → 1229.6).
- **GPU-efficiency references** (official champion tok/s @ cumulative GPU-min of the run):
  - `v3/framework-eval`: 707.6 @56, 757.6 @95, 843 @223, 880 @275, 907 @378, 951.5 @476.
  - `v4/sue`: 728.8 @69, 760.8 @92, 785.3 @220, 838 @295, 897.5 @404, 908.5 @462.
  - `v4/sse`: 712.1 @94, 761.6 @111, 835.1 @158 (stopped by the human).

## Map (families; ✓ win, ~ small/noise, ✗ dead end; the gain is the typical best single measurement)
```
A. TP communication (no P2P, F1)
   ✓ host-ar ................. +5%    one-shot AR via pinned host memory for small (decode) ARs
     ✓ l2-prefetch-in-ar ..... +2-5%  AR kernel's spare blocks pull the next GEMM's weights into L2
     ✓ ar-compression ........ +1.9%  lossless bf16 payload code (fewer PCIe bytes)
     ✓ numa-bind ............. +0.7%  TP workers + AR buffers on the GPU's NUMA node
     ~ host-allgather ........ ~0     host-staged logits all-gather (neutral building block)
     ✗ ar-kernel-tuning, ar-norm-fusion, decode-ar-gemm-overlap, ce-decode-ar
   ✓ mixed-ar-hiding ......... +1-3%  token-split chunks + copy-engine AR lanes in prefill steps
   ✗ nccl-env, flashinfer-pcie-ar
B. Drafter (MTP speculative decoding; verification exact)
   ✓ fp8-draft-head .......... +3%    FP8 copy of lm_head for the drafter + local draft argmax
     ✓ draft-vocab-subset .... +0.5-1.4%  drafter head over the first N (frequent) token ids
       ✓ draft-head-screen ... +1.1%  PCA-screened draft head
     ✓ drafter-package ....... +1.1%  fc table, packed argmax, glue tiles
   ✓ draft-depth-3 ........... +4.9% on gdn-two-slot; harmful without it (F5)
   ✗ trained-draft-head (also n-gram drafting, budgeted depth)
C. Target sampling / lm_head
   ✓ greedy-fastpath ......... +1.5-2.5%  exact argmax over vocab shards (alt: sharded-sampling)
   ✓ sharded-sampling ........ +1.4-1.8%  stock flag
   ✓ target-lm-head-bytes .... +1.8%, +0.7%  lossless codec / certified exact screen of the target lm_head
D. GEMMs (FP8 weights run on Marlin weight-only, F7)
   ✓ w8a8-mixed .............. +4.7-8.4%  FP8 activations (checkpoint scheme) for M>96 only
     ✓ w8a8-fused-quant ...... +1-1.6%  activation quant fused into norm / SiLU producers
     ✓ w8a8-gate-up-mid-m .... +2.7% (k=3, M=64)
     ✓ e4m3-decode-gemm ...... +2.5%  register-direct FP8 decode GEMM (needs l2-prefetch-in-ar)
   ~ marlin-tuning ........... +0.5-0.9%  tile configs / slab policy
   ✗ triton-linear (all M), dequant-cublas-prefill, custom-decode-gemm
E. GDN state (spec decode)
   ✓ gdn-replay .............. +2.6-4.4%  write 1 state/step + replay accepted tokens
     ✓ gdn-lazy-commit ....... +1.0-1.6%
     ✓ gdn-two-slot .......... enables draft-depth-3 at 16 requests (see B)
     ✓ gdn-mixed-and-kernel .. +0.5-1.3% each  replay on mixed path, RED8, batched prologue
F. Host overhead / graphs
   ✓ prefill-graphs .......... +2.3-5.5%  CUDA graphs for prefill-containing steps (memory trade, F13)
   ✓ attn-host-plan .......... +2-5.5%  remove FlashInfer's per-step host planning (TRITON_ATTN or host-bound plan)
   ~ stock-flag-tweaks
G. Mixed-step scheduling
   ✓ mixed-waves ............. +2.6%  2-wave layer pipelining + AR link order + captured GDN prefill core
   ✗ spatial-overlap (also PP=2, megakernels)
```

## Graph (dependencies)
Generated from the nodes' `builds_on` (`python3 .claude/skills/eval-run-qwen38/keygraph.py <this file>`).
<!-- graph:start -->
```
BASE
├─ host-ar  [win]
│  ├─ l2-prefetch-in-ar  [win on host-ar]
│  ├─ ar-compression  [win on host-ar]
│  ├─ numa-bind  [win on host-ar]
│  ├─ host-allgather  [no gain]
│  ├─ ar-kernel-tuning  [no gain]
│  ├─ ar-norm-fusion  [no gain]
│  ├─ decode-ar-gemm-overlap  [no gain]
│  └─ ce-decode-ar  [no gain]  alt: host-ar
├─ mixed-ar-hiding  [win (with copy engines on host-ar); no gain with SM-based NCCL overlap]
│  └─ mixed-waves  [win]
├─ nccl-env  [no gain]
├─ flashinfer-pcie-ar  [doesn't run]  alt: host-ar
├─ fp8-draft-head  [win]
│  └─ draft-vocab-subset  [win (small)]
│     └─ draft-head-screen  [win (small)]
│        └─ drafter-package  [win (small)]
├─ draft-depth-3  [harmful on stock GDN state; win on gdn-two-slot]
├─ trained-draft-head  [no gain]
├─ greedy-fastpath  [win]  alt: sharded-sampling
│  └─ target-lm-head-bytes  [win (small)]
├─ sharded-sampling  [win]  alt: greedy-fastpath
├─ w8a8-mixed  [win]
│  ├─ w8a8-fused-quant  [win (small)]
│  ├─ w8a8-gate-up-mid-m  [win]  + draft-depth-3
│  └─ e4m3-decode-gemm  [win]  + l2-prefetch-in-ar
├─ marlin-tuning  [win (small)]
├─ triton-linear  [harmful]
├─ dequant-cublas-prefill  [no gain]
├─ custom-decode-gemm  [no gain]
├─ gdn-replay  [win]
│  ├─ gdn-lazy-commit  [win (small)]
│  ├─ gdn-two-slot  [win (as the enabler of draft-depth-3)]
│  └─ gdn-mixed-and-kernel  [win (small)]
├─ prefill-graphs  [win (more on host-ar)]
├─ attn-host-plan  [win]
├─ stock-flag-tweaks  [no gain]
└─ spatial-overlap  [no gain]
```
<!-- graph:end -->

## Node format
```
### `id`: title
- builds_on: BASE | id [+ id]   the changes it needs (its result depends on them)
- alternative_to: id            optional: substitutes (same effect, don't stack); either counts
- status: win | win on <id> | no gain | harmful | doesn't run
- what / recognize              enough to spot it in a run, and what it is not
- results:                      one line per base: `on <stack>: <result> <strength> (evidence)`
- best:                         the best known realization (what to compare a run's version with)
- cheapest path:                the least GPU time known to get a trustworthy end-to-end number
- traps:                        known ways runs got it wrong (feeds evals/failure_cases.md)
- cause / revisit / grading     for dead ends: why, and what would make it worth retrying
```

## A. TP communication
Each decoder layer ends its mixer (`out_proj` / `o_proj`) and its MLP (`down_proj`) with a row-parallel
GEMM whose partial sums are all-reduced: 128 ARs + 1 (embedding) per forward, ~0.5 MB each at decode
(48 rows × 5120 bf16), MBs in prefill-containing steps. No P2P (F1), so every byte goes GPU → host
RAM → GPU. In-engine NCCL (LL protocol) ≈ 72 µs per decode AR on a 2026-11 stack, ~8-10 ms of a
~39 ms decode step (F17).

### `host-ar`: one-shot host-staged 2-rank all-reduce for small tensors
- builds_on: BASE
- status: win
- what: pinned host memory (shared file mmap'd by both TP workers, `cudaHostRegister` mapped) gives
  each GPU a device pointer. One kernel: write own partial → `__threadfence_system` → flag (sequence
  number) → spin on the peer's flag (uncached loads) → read the peer's partial → fp32 add → bf16.
  Bitwise equal to NCCL. Only below a size cutoff (~1-1.3 MB, ≤ ~128 rows); NCCL above.
- recognize: new AR backend in `vllm/distributed/device_communicators/` dispatched before PyNCCL for
  TP=2; pinned/mapped host memory, system fences, flag polling. Not this: NCCL env tuning
  (`nccl-env`); vLLM's stock custom AR (needs P2P); copy-engine ARs (`mixed-ar-hiding`).
- results:
  - on BASE: +5.0% official (`v1/test-3:T1-002`; `v2/test-6:T1-002`), +5.7% official (`v1/test-4:T1-001`).
  - on sharded-sampling: +4.8% official (`baseline/run-1:exp003`).
  - on fp8-draft-head: +4.35% `abab`; with it 707.6 official, +11.3% vs BASE (`v3/framework-eval:T1-002`, T1-004).
  - on fp8-draft-head + sharded-sampling: +3.4% `unofficial` (`v4/sue:T1-002`); rebased on + prefill-graphs:
    +5.1% `unofficial` (`v4/sue:T1-005`).
  - per call: 61-72 → 43-50 µs at 48 rows; floor ~34-40 µs (F3).
- best: fixed grid (e.g. 4 blocks) so the sequence parity is uniform across calls; per-block flags;
  NUMA-local buffer (`numa-bind`); cutoff ~1-1.3 MB.
- cheapest path: a small kernel + one in-server measurement (~5 GPU-min unofficial). framework-eval
  had it in its first official champion at 56 GPU-min.
- traps:
  - judging it by a standalone microbenchmark against forced NCCL Simple (57 µs) instead of the
    engine's NCCL (~72 µs), and calling 57 µs "the floor" (`v4/sse:T1-003`, evals/failure_cases.md).
  - timing on one rank without per-sample sync: rank skew inflates and scatters the number
    (52 vs 92 µs for the same kernel, `v4/sse:T1-003`).
  - counting "loses at M ≥ 96" against it: the cutoff routes those to NCCL.
  - cross-call parity race when the grid varies (`v4/sue:T1-002`, fixed in review).

### `l2-prefetch-in-ar`: prefetch the next GEMM's weights into L2 during the all-reduce
- builds_on: host-ar
- status: win on host-ar
- what: during the PCIe-bound AR (~45 µs) HBM is idle. Extra blocks of the AR kernel (or a side
  stream) issue `evict_last` prefetches of the next GEMM's FP8 weights into the 72 MB L2; the next
  Marlin GEMM (DRAM-bound, F7) then reads part of its weights from L2.
- recognize: prefetch range(s) passed to the AR call; L2 persisting / evict_last loads of the next
  layer's weight. Not this: GEMM‖AR overlap (`decode-ar-gemm-overlap`).
- results:
  - on test-6's T1-016 stack: +3.9% official (`v2/test-6:T1-019`, KB25); + out_proj/o_proj and
    lm_head-prefix ranges +1.6% official (T1-020); size gate 1 MiB → 2.7 MB +0.3% official (T58-001).
  - on framework-eval's stack: +2.0% official (`v3/framework-eval:T3-003`); fixing the deferred-AR case
    +1.4% and +1.3% official (T3-005, T3-006).
  - on v4/sue's stack: side-stream variant +2.3% `unofficial` (`v4/sue:T2-005`); later variants -1.5..-1.8%.
- traps: with a deferred AR the prefetch silently never ran (`v3/framework-eval:T3-005`); denser or
  earlier-stopping prefetch variants lose in-server (`v2/test-6:T1-021`).

### `ar-compression`: lossless compression of the AR payload
- builds_on: host-ar
- status: win on host-ar
- what: per 32 bf16 values a shared exponent header + shorter codes; decoded bitwise; fewer PCIe
  bytes for decode and prefill-size host ARs.
- results: +1.9% official, same-job +2.0% (`v2/test-6:T1-030`, KB43). On the copy-engine lane: -1.4%/layer
  `component` (T1-031); CE payload codec slower (T9-007).

### `numa-bind`: NUMA self-bind of TP workers
- builds_on: host-ar
- status: win on host-ar
- what: pin each TP worker (and so its pinned AR buffers) to the GPU's NUMA node at init.
- results: +0.7% official and CV 1.8% → 0.4% (`v2/test-6:T1-026`); cause: AR slots on the far node
  in 7/10 unbound starts (KB37).
- traps: unexplained bimodal starts are a symptom of this, not noise.

### `host-allgather`: host-staged all-gather for the logits
- builds_on: host-ar
- status: no gain
- results: ~0 `ab` (`v1/test-3:T1-004`; `v2/test-6:T1-004`). Kept as a neutral exact building block.
- note: `fp8-draft-head` (local argmax) and `greedy-fastpath` remove most logits all-gathers.

### `ar-kernel-tuning`: micro-tuning the host AR kernel
- builds_on: host-ar
- status: no gain
- results: chunk pipelining loses (`v1/test-3:KB5`); unified blocks / per-warp flags null (`v1/test-4:T1-002`);
  grid 4..32 flat (`v4/sue:T3-001`); cut-through chunks -1.2% (`v2/test-6:T36-002`); fence-free LL AR 2-3 µs/AR
  (`v2/test-6:T20-001`); decode AR below its PCIe cost +0.17% official, noise (`v3/framework-eval:T2-012`).
- cause: host-ar is within ~5-10 µs of the PCIe floor (F3).

### `ar-norm-fusion`: fuse residual add + RMSNorm into the AR kernel
- builds_on: host-ar
- status: no gain
- results: ~1.5-3 µs/call `component` (`baseline/run-1:tsv:11`); +0.26% `abab`, noise
  (`v3/framework-eval:T1-011`); -1.6% screen (`v4/sue:T1-007`); -0.6% `abab` (`v2/test-6:T1-043`).

### `decode-ar-gemm-overlap`: hide the decode AR under GEMMs (chunked GEMM+AR, producer-fused AR)
- builds_on: host-ar
- status: no gain
- results: N-chunked GEMM+AR ~1% at best `component` (`v2/test-6:T1-018`); overlap saves ≤ 5-8 µs/AR
  (`v2/test-6:T4-001`); producer-fused AR: posted writes slow the GEMM (`v2/test-6:T13-001`); overlapped
  GEMM+AR slower than serial (`v3/framework-eval:T1-007`, T2-005); prefill slabs in AR windows +20-33%
  (`v2/test-6:T11-001`).
- cause: at 0.5 MB the AR is short and the GEMM uses all SMs (F9); extra launches cost more than
  the overlap saves.

### `ce-decode-ar`: copy-engine or hybrid CE/SM transport for decode-size ARs
- builds_on: host-ar
- alternative_to: host-ar
- status: no gain
- results: decode CE AR +23 µs/AR vs host-ar (`v2/test-6:T9-006`); hybrid +11 µs/AR, CE start latency
  ~20 µs in graphs (`v2/test-6:T16-001`, KB74).

### `mixed-ar-hiding`: hide prefill-step all-reduces behind compute
- builds_on: BASE
- status: win (with copy engines on host-ar); no gain with SM-based NCCL overlap
- what: split a prefill-containing step's rows into 2 token chunks; chunk A's AR runs on the copy
  engines (D2H + H2D through pinned slots, pipelined in pieces) while chunk B computes. Copy engines
  don't take SMs from the GEMM being overlapped (NCCL / host-ar kernels do, F9).
- recognize: token- or row-split of mixed steps with ARs on a side stream; DMA-staged AR. Not this:
  decode-size overlap (`decode-ar-gemm-overlap`).
- results:
  - SM-based (NCCL) overlap: 0.0% ± 0.7% `ab` (`v1/test-4:T1-004`); DBO microbench ~50% hidden, net ~0
    (`v4/sse:T1-004`).
  - CE-based, on test-6's stack: hybrid split for steps ≥ 256 tokens +1.5% official (+0.7% same-job,
    `v2/test-6:T1-025`); DMA lane in 4 pieces +0.9% official (T1-042); in-segment CE overlap +0.7% official
    (T9-005).
  - row-split on framework-eval's stack: +1.1% `abab` (`v3/framework-eval:T1-013`); restored after a fused
    norm disabled it: +2.1% official (T3-009). Chunked GEMM+AR in eager steps +0.5% `unofficial` (`v4/sue:T1-012`).
- traps: a later fusion silently switched it off (`v3/framework-eval:T3-008`/T3-009).

### `nccl-env`: NCCL env knobs (protocol, channels, buffers)
- builds_on: BASE
- status: no gain
- results: no gain `component` (`v1/test-4:KB1`; `baseline/run-1:tsv:9`); +0.3% `ab` noise (`baseline/run-1:tsv:21`);
  `NCCL_PROTO=Simple` applied but in-engine NCCL time unchanged, +1.0% was noise (`v4/sse:T1-002`, reverted);
  knob sweep null (`v4/sse:T1-005`).
- cause: standalone gains (Simple 1.3-2× faster than LL at some sizes) don't appear in the engine.
- traps: inferring the protocol from kernel names is wrong; check the NCCL debug log (`v4/sse:T1-002`).

### `flashinfer-pcie-ar`: `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`
- builds_on: BASE
- alternative_to: host-ar
- status: doesn't run
- results: fails to initialize without P2P (`baseline/run-1:tsv:4`; `v4/sse` jobs 3, 10).

## B. Drafter (MTP)
MTP proposes k draft tokens per step; the target verifies them in one pass (exact). The drafter's
cost per draft step is dominated by its lm_head (248,320 × 5120, ~1.27 GB bf16 per full head).

### `fp8-draft-head`: FP8 drafter lm_head + local draft argmax
- builds_on: BASE
- status: win
- what: spec config `"draft_lm_head_quantization": "fp8"` (drafter-only FP8 copy; the target keeps
  bf16) and `"use_local_argmax_reduction": true` (per-shard argmax + small reduce instead of a
  logits all-gather). Acceptance unchanged.
- recognize: those spec-config keys or equivalent code. Old id: `draft-head-diet`.
- results: +2.4% `ab` on BASE (`v1/test-4:T2-001`); +3.0% `ab` on host-ar + prefill-graphs (`v1/test-3:T1-007`;
  `v2/test-6:T1-007`); +3.26% `unofficial` on BASE (`v4/sue:T1-001`); with host-ar +11.3% official
  (`v3/framework-eval:T1-004`).
- cheapest path: serve flags only; one unofficial eval.
- traps: +~0.6 GB/rank. On a stack whose graphs/KV already use the headroom it costs running
  requests (15 of 16 with a 4 GiB KV, F13) and loses (`v4/sse:T2-002`): the fix is the memory budget,
  not dropping the head.

### `draft-vocab-subset`: drafter lm_head over the first N token ids
- builds_on: fp8-draft-head
- status: win (small)
- what: low BPE ids are the frequent tokens; the drafter's head covers ids [0, N) (re-sharded, FP8).
  Drafts stay in the subset; the target verifies over the full vocab (exact).
- results: N=98304 +2.1% `ab`, official within noise (`baseline/run-1:exp008`); +1.0% `abab` (`v2/test-6:T1-013`);
  +1.4% official (`v3/framework-eval:T2-011`); +0.5% `unofficial` (`v4/sue:T2-011`); N=65536 + local argmax
  + mbt 1024 +1.3% `unofficial` (`v4/sse:T2-004`). N=64K loses acceptance in run-1 (`tsv:18`).

### `draft-head-screen`: PCA-screened draft head
- builds_on: draft-vocab-subset
- status: win (small)
- what: a cheap low-rank screen picks candidate rows; only those rows of the FP8 head are evaluated.
- results: +1.1% official (`v2/test-6:T27-002`, KB84).

### `drafter-package`: drafter fixed costs
- builds_on: draft-head-screen
- status: win (small)
- what: precomputed embedding half of the MTP fc (table in pinned host memory), packed (value,
  index) argmax all-gather, fewer blocks for mamba glue kernels.
- results: +1.1% official, +1.7% `abab` (`v2/test-6:T57-004`).

### `draft-depth-3`: 3 MTP draft tokens
- builds_on: BASE
- status: harmful on stock GDN state; win on gdn-two-slot
- results:
  - stock state: -3.9% to -6.9% `ab`, TTFT gate fails (`v1/test-4:T2-002`; `v1/test-3:T1-008`; `v2/test-6:T1-008`);
    -4.9% `unofficial` (`v4/sue:T1-003`); 602.5 tok/s, gates fail (`v4/sse:T1-001`); more KV OOMs (`v1/test-3:T1-010`).
  - on gdn-two-slot: +4.9% official (1142.2 vs 1088.9, `v2/test-6:T37-002`).
- cause: each request holds 1+k GDN state blocks (F5), so at k=3 only ~14 of 16 requests fit:
  admission stalls, TTFT explodes. With two-slot state all 16 run.
- grading: a correct drop cites capacity/admission (F5); "step cost" alone is incomplete.

### `trained-draft-head`: fine-tuned / distilled MTP head; n-gram drafting; budgeted depth
- builds_on: BASE
- status: no gain
- results: fine-tunes no held-out gain (`v2/test-6:T3-002..004`); n-gram drafting E +0.7-2.7% only
  (T12-001); budgeted K_MAX=3 -2.6% (T27-001); trees / self-distilled heads withdrawn (T38).

## C. Target sampling / lm_head

### `greedy-fastpath`: exact greedy argmax over vocab shards for verification
- builds_on: BASE
- alternative_to: sharded-sampling
- status: win
- what: per-shard (max, argmax), all-gather of [rows, 2], cross-rank tie-break equal to the full
  argmax (F10); only for pure greedy batches; full path otherwise.
- results: ~+1.5-2% official (`v1/test-3:T1-009`; `v2/test-6:T1-009`); ~+1.2% (`v3/framework-eval:T1-012`);
  +2.1% `unofficial` (`v4/sue:T3-004`); +2.5% `unofficial`, in an official +12.0% stack (`v4/sse:T1-006`).
  Tie with sharded-sampling (`v1/test-4:T2-006`; `baseline/run-1:tsv:13`).

### `sharded-sampling`: `--enable-batch-sharded-sampling`
- builds_on: BASE
- alternative_to: greedy-fastpath
- status: win
- results: +1.4% official on BASE (`baseline/run-1:exp002`); +1.8% `abab` (`v3/framework-eval:T1-005`);
  +1.6% `unofficial` (`v4/sue:T1-004`). Costs a SendRecv ~0.41 ms/step (F11). Disables test-6's greedy path.

### `target-lm-head-bytes`: fewer bytes for the target lm_head, exactly
- builds_on: greedy-fastpath
- status: win (small)
- what: lossless codec of the bf16 target lm_head decoded inside a custom GEMM (`v2/test-6:T1-037`), or a
  certified screen (7-bit plane + error bound) that falls back to exact rows (T42-003).
- results: codec +1.8% official (T1-037; v2 no gain, T1-038); certified screen in a +0.7% official bundle (T50-002).

## D. GEMMs
Weights are FP8 128×128 block-scaled, but sm_89 runs them through Marlin (weight-only, bf16 MMA,
F7). Decode GEMMs are DRAM-bound; prefill GEMMs (M > ~96) are compute-bound.

### `w8a8-mixed`: FP8 activations for large-M (prefill/mixed) GEMMs
- builds_on: BASE
- status: win
- what: quantize activations to e4m3 with the checkpoint's own scheme (dynamic, per-token group of
  128, fp32 accumulation across K-blocks) and use FP8 MMA (2× bf16 throughput) for M > ~96; decode
  (M ≤ 96) stays on Marlin. Allowed by the 2026-11-09 activation ruling; needs the GSM8K gate (F14).
- recognize: W8A8 dispatch by M (Triton block-FP8 kernel, a Marlin-layout FP8 GEMM, or vLLM's
  `humming` backend). Not this: `triton-linear` (all M, decode too).
- results: Marlin-layout W8A8 for M > 96: +8.1% official, +9.2% `abab` (`v2/test-6:T29-004`); Triton W8A8:
  +4.7% `unofficial`, in a +4.3% official stack (`v3/framework-eval:T2-004`, T2-006); W8A16/W8A8 Triton +3.0%
  `unofficial` (`v4/sue:T1-016`, T1-017); humming W8A8 +8.4% `unofficial` (`v4/sse:T2-005`, official via
  T2-006 761.6).
- traps: humming's JIT writes `~/.humming` unless redirected (task env fixes it).

### `w8a8-fused-quant`: activation quant fused into its producers
- builds_on: w8a8-mixed
- status: win (small)
- results: norm+quant / SiLU+quant fusion +1.1% official (`v3/framework-eval:T2-009`); in a +1.6% official
  bundle with configs + GDN glue (`v2/test-6:T29-006`).
- traps: the fusion disabled row-split AR overlap (`v3/framework-eval:T3-008`).

### `w8a8-gate-up-mid-m`: W8A8 for gate_up at 48 < M ≤ 96
- builds_on: w8a8-mixed + draft-depth-3
- status: win
- results: +2.7% official, +3.8% `abab` at k=3 (M=64) (`v2/test-6:T45-005`).

### `e4m3-decode-gemm`: register-direct FP8 decode GEMM
- builds_on: w8a8-mixed + l2-prefetch-in-ar
- status: win
- results: +2.5% official (`v2/test-6:T31-003`); cashes in the L2-resident weights from the AR prefetch.

### `marlin-tuning`: Marlin tile configs / slab policy for decode batches
- builds_on: BASE
- status: win (small)
- results: {64,128,128} for 2-3 m-blocks +0.6% official (`baseline/run-1:exp005`), +0.45% noise
  (`v2/test-6:T1-016`); {128,64,128} for M 17-48 +0.85% `unofficial` (`v4/sue:T2-012`); slab policy M 72-120
  +0.8% official (`v2/test-6:T14-004`).
- note: Marlin is near its DRAM floor (F7), so kernel-level gains are small.

### `triton-linear`: `--linear-backend triton` for all M
- builds_on: BASE
- status: harmful
- results: TTFT -14% but decode ITL +14%, net -4% (`qwen38_opt`).
- revisit: done: the M-split version is `w8a8-mixed`.

### `dequant-cublas-prefill`: dequantize + cuBLAS for large M
- builds_on: BASE
- status: no gain
- results: large-M Marlin = cuBLAS bf16 `component` (`baseline/run-1:tsv:12`).

### `custom-decode-gemm`: hand-written decode GEMMs replacing Marlin
- builds_on: BASE
- status: no gain
- results: 2-warps/scheduler, swap-AB register-direct, L2-warm variants all slower than Marlin
  (`v2/test-6:T1-029`, T8-003, T24-001, T26-001..003). The FP8 one wins only with W8A8 + L2 prefetch
  (`e4m3-decode-gemm`).

## E. GDN state (speculative decoding)
48 GDN layers carry a 128×128 fp32 recurrent state per head. Stock MTP decode stores k+1 full
states per step (one per draft position) and each request holds 1+k state blocks (F5).

### `gdn-replay`: checkpoint + replay of accepted tokens
- builds_on: BASE
- status: win
- what: store only the committed state + a log of the step's token inputs; next step replay the
  accepted tokens with the same fp32 ops in the same order (bitwise); a flush materializes the stock
  layout before anything else reads it.
- recognize: custom GDN decode kernel in `csrc/` storing 1 state/step + a log, plus a flush path.
- results: +2.6% official (`v1/test-3:T1-012`; `v2/test-6:T1-012`); +4.4% official (`v3/framework-eval:T2-008`);
  +3.5% `unofficial` after a fix (`v4/sue:T2-016`); +4.0% `unofficial`, official +9.7% over the
  previous official champion together with attn-host-plan (`v4/sse:T2-008`).
- traps: replay gated wrongly → silent no-op (`v4/sue:T2-015`, fixed in T2-016).

### `gdn-lazy-commit`: skip most committed-state stores
- builds_on: gdn-replay
- status: win (small)
- results: +1.0% official (`v2/test-6:T1-035`); +1.6% `unofficial` (`v4/sue:T2-018`).

### `gdn-two-slot`: two state slots per request regardless of k
- builds_on: gdn-replay
- status: win (as the enabler of draft-depth-3)
- what: committed state + replay log only, instead of 1+k slots, so k=3 fits 16 requests.
- results: with k=3: +4.9% official (`v2/test-6:T37-002`); two-slot alone bitwise, an asset (`v2/test-6:T5-005`).

### `gdn-mixed-and-kernel`: replay on the mixed path; GDN kernel micro-work
- builds_on: gdn-replay
- status: win (small)
- results: replay on mixed steps + every-32 graph grid +1.3% official (`v2/test-6:T1-023`); RED8 reduction
  +0.7% official (T1-040); batched prologue +0.5% official (T1-041). State codec slower (T1-039).

## F. Host overhead / CUDA graphs

### `prefill-graphs`: CUDA graphs for prefill-containing steps
- builds_on: BASE
- status: win (more on host-ar)
- what: `--compilation-config '{"max_prefill_cudagraph_capture_tokens": N}'` (BASE graphs only steps
  ≤ 96 tokens, so prefill steps run eager and launch-bound).
- results: on BASE: +2.2% `ab` noise (`v1/test-3:T1-001`; `v2/test-6:T1-001`); +2.3% `unofficial` at N=2048
  (`v4/sse:T2-001`); +4.5% `unofficial` at N=512 on fp8-draft-head + sharded-sampling (`v4/sue:T2-004`). On
  host-ar: +5.5% official (`v2/test-6:T1-005`), super-additive (KB4).
- traps: graph memory competes with KV and other memory users (F13). N=2048 took the headroom the
  FP8 draft head needed in `v4/sse` (T2-002 lost to 15 of 16 requests); test-6 capped 1536 when 2048
  OOMed (T9-005). Pick N with the memory budget in view.

### `attn-host-plan`: remove FlashInfer's per-step host planning from the decode critical path
- builds_on: BASE
- status: win
- what: with FP8 KV the default backend is FlashInfer, whose per-step CPU planning (incl. a blocking
  `seq_lens.cpu()`) leaves the GPU idle ~1-1.4 ms before each decode graph (F12). Either switch to
  `--attention-backend TRITON_ATTN`, or plan FlashInfer from a host upper bound + exact GPU metadata.
- recognize: the flag, or a change that removes that sync (target or drafter side). Not this: Triton
  attention on a base without FP8 KV (`stock-flag-tweaks`). Old id: `triton-attn`.
- results: TRITON_ATTN +6.8% official (`baseline/run-1:exp006`), +5.5% `abab` (`v2/test-6:T1-014`), +3.8%
  official (`v3/framework-eval:T1-009`); host-bound plan +1.9% + split-KV +1.3% `unofficial` (`v4/sue:T1-010`,
  T1-015); drafter-only host-bound plan +2.3% `unofficial` (`v4/sse:T1-008`).
- cheapest path: one serve flag + one unofficial eval. Found by graph-level nsys (node-level traces hide it, F8).

### `stock-flag-tweaks`: common serve flags
- builds_on: BASE
- status: no gain
- results: prefix caching off, GDN backend flags, mbt 4096/8192 within noise (`qwen38_opt`;
  `baseline/run-1:tsv:24`); `--no-async-scheduling` -10%. Prefix caching off is forbidden (TASK.md, 2026-11-15).

## G. Mixed-step scheduling

### `mixed-waves`: layer-pipelined waves for big mixed steps
- builds_on: mixed-ar-hiding
- status: win
- what: steps ≥ 768 tokens run as 2 waves without a per-layer join; chunk 1's ARs wait for chunk 0's
  (link order on the single D2H copy engine); GDN prefill core captured so waves aren't host-bound.
- results: +2.6% official, +2.8% `abab` (`v2/test-6:T52-005`); eager 2-wave prototype -7.5% (host-bound,
  T30-004).

### `spatial-overlap`: run prefill and decode concurrently on split SMs; PP=2; megakernels
- builds_on: BASE
- status: no gain
- results: green-context co-run zero-sum (`v2/test-6:T2-001..004`, KB59); PP=2 decode loses (T38-005);
  decode megakernel at parity / ceiling too low (T52-001/002, T53).

## Facts (cited by nodes; used to judge whether a run's reasons were right)
- **F1.** No GPU P2P: vLLM's custom AR is disabled; TP AR is NCCL through host memory over PCIe.
- **F2.** Box state (2026-11): GPU 0's SM clock locked at 2235 MHz, GPU 1 boosts higher; compute-bound
  kernels ~15-17% slower on GPU 0 (prefill AR skew) (`v1/test-3:KB8`). Check gpu_clocks.csv.
- **F3.** AR floor: 0.5 MB through host at ~13 GB/s of AR payload (each byte crosses PCIe twice) ≈
  34-40 µs at 48 rows; SM-issued PCIe is not full duplex (`v1/test-3:KB5`; `v2/test-6:T1-017` 21+21 µs, KB23).
  Large ARs are bandwidth-bound for every transport: host-ar = NCCL above ~2.5 MB.
- **F4.** Decode step anatomy on BASE (~35-40 ms): Marlin ~50%, AR 6-10 ms, lm_heads ~4.3 ms, logits
  all-gathers ~1.7 ms, GDN ~2.4-3.7 ms + L2 spill. Prefill-containing steps ~36% of wall time.
- **F5.** GDN spec-state blocks per request = 1+k per GDN group (10 at k=2, 13 at k=3), shared with the
  KV pool; at k=3 only ~14 of 16 requests fit.
- **F6.** KV above ~5.0 GiB/rank OOMs on test-3's stack; the budget depends on graphs and other buffers.
- **F7.** FP8 weights run through Marlin (weight-only, bf16 MMA) on sm_89; decode GEMMs DRAM-bound at
  ~80-92% of peak.
- **F8.** Measurement: one server per arm drifts ~2%; close calls need interleaved fresh starts.
  Node-level nsys inflates graph launches; host stalls show only at graph level.
- **F9.** Marlin's persistent grid occupies all SMs and blocks concurrent SM kernels (NCCL, host-ar).
- **F10.** bf16 ties for the max in ~1.8% of rows: exact argmax must break ties like the stock path.
- **F11.** Batch-sharded sampling adds a SendRecv of ~0.41 ms per decode step.
- **F12.** With FP8 KV the default attention is FlashInfer; its per-step host planning idles the GPU
  ~0.9-1.4 ms before each decode graph.
- **F13.** KV capacity is quantized by the hybrid page (~1,600 tokens); a 4 GiB KV pool admits 15 of 16
  requests. Graph capture size, extra heads and AR buffers all compete for the same headroom
  (`v4/sse:reports/final.md`, T2-002).
- **F14.** W8A8 (FP8 activations) is allowed only in the checkpoint's scheme (dynamic group-128 e4m3,
  fp32 across K-blocks) and must pass the GSM8K gate (ruling 2026-11-09).
- **F15.** Standalone microbenchmarks can disagree with the engine (rank skew, other traffic, graph
  launch); in-engine NCCL is slower than standalone NCCL at decode size.
- **F16.** v4 unofficial evals (200 prompts) understated official gains ~2× for prefill-heavy changes
  (`v4/sse`: T1-006 +6.2% unofficial vs +12.0% official over BASE).
- **F17.** In-engine NCCL decode AR ≈ 72 µs (9.66 ms per step / ~135 calls at 48 rows,
  `v4/sse:reports/T1-sweep/update-2.md:8`).

## History
- 2026-11-07: created from v1 test-3, v1 test-4, qwen38_opt; DAG format; baseline/run-1 added (full
  earlier entries in git history).
- 2026-11-15: rebuilt around families from v2/test-6 (best run, 1229.6), v3/framework-eval, v4/sue and
  v4/sse. New nodes: l2-prefetch-in-ar, ar-compression, numa-bind, decode-ar-gemm-overlap, ce-decode-ar,
  mixed-ar-hiding (absorbs prefill-ar-overlap and ce-ar-prefill), draft-head-screen, drafter-package,
  trained-draft-head, target-lm-head-bytes, w8a8-mixed, w8a8-fused-quant, w8a8-gate-up-mid-m,
  e4m3-decode-gemm, custom-decode-gemm, gdn-lazy-commit, gdn-two-slot, gdn-mixed-and-kernel,
  attn-host-plan (absorbs triton-attn), mixed-waves, spatial-overlap. Renamed draft-head-diet →
  fp8-draft-head. Corrected: prefill-graphs pays on BASE too (+2.3-4.5%, v4); draft-depth-3 is a win on
  gdn-two-slot. New fields: best, cheapest path, traps. Facts F13-F17. Frontier 749.7 → 1229.6.
