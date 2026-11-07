# Answer key: qwen38-27b-fp8-2x4090

A map of this task's search space, as far as we know it: a DAG of changes, each with what is known
about it on each base (works, doesn't and why, or untested), plus the facts that explain them.
Merged and curated from all runs of any flavor; not a copy of any run's graph. A lower bound, and it
evolves: format and update rules are in `EVAL.md` ("The answer key").

Metric: output tok/s at c=16 (TASK.md). BASE ≈ 631-636 tok/s official (varies ~1% between runs).
Evidence ids: `<flavor>/<run>:<record, KBn, or path:line>` (e.g. `v1/test-3:KB13`); `qwen38_opt` =
the hand-driven study in `../qwen38_opt/CLAUDE.md` (its BASE differed: 3 draft tokens, no FP8 KV).
Evidence strength, strongest first: `official` (fresh-server trials), `abab` (interleaved same-job
A/B), `ab` (one same-job A/B), `component` (microbench / nsys of one part), `estimate`.

## Frontier (best known)
`host-ar` + `prefill-graphs` + `host-allgather` + `draft-head-diet` + `greedy-fastpath` +
`gdn-replay` = **749.7 tok/s official, +18.7% vs BASE 631.4** (`v1/test-3:T1-012`).

## Graph
Generated from the nodes' `builds_on` (`python3 .claude/skills/eval-run/keygraph.py <this file>`):
each node under its first parent, other parents as "+ X", alternatives as "alt: X". Regenerate it
after every edit.
<!-- graph:start -->
```
BASE
├─ host-ar  [win]
│  ├─ prefill-graphs  [win on host-ar; no gain on BASE]
│  ├─ host-allgather  [no gain]
│  ├─ ar-kernel-tuning  [no gain]
│  ├─ ar-norm-fusion  [open]
│  ├─ l2-prefetch  [open]
│  └─ ar-compression  [open]
├─ draft-head-diet  [win]
│  └─ reduced-vocab-draft-head  [ruled out by estimate]
├─ greedy-fastpath  [win]  alt: sharded-sampling
├─ sharded-sampling  [win]  alt: greedy-fastpath
├─ gdn-replay  [win]
│  └─ small-spec-state  [open]
│     └─ draft-depth-3-small-state  [open]  + draft-depth-3
├─ draft-depth-3  [harmful]
├─ nccl-env  [no gain]
├─ prefill-ar-overlap  [no gain measurable]
├─ triton-linear  [harmful]
├─ stock-flag-tweaks  [no gain]
├─ pp2  [ruled out by estimate]
├─ uneven-tp  [ruled out by estimate]
├─ dequant-cublas-prefill  [ruled out by estimate]
├─ spec-glue-graph  [open]
├─ marlin-tuning  [open]
├─ ce-ar-prefill  [open]
└─ flashinfer-pcie-ar  [open]  alt: host-ar
```
<!-- graph:end -->

## Node format
```
### `id`: title
- builds_on: BASE | id [+ id]   the changes it needs (its result depends on them). Not the same as
                                the stack it happened to be measured on, which goes in results.
- alternative_to: id            optional: substitutes (same effect, don't stack); either counts.
- status: win | win on <id> | no gain | harmful | ruled out by estimate | open   (summary)
- what / recognize              enough for a judge to spot it in a run by what the change does,
                                and what it is not.
- results:                      one line per base: `on <stack>: <result> <strength> (evidence)`
- cause / revisit / grading     why it fails, what would make it worth retrying, judging hints.
```

## Nodes

### `host-ar`: host-staged 2-rank all-reduce (no P2P)
- builds_on: BASE
- status: win
- what: a custom all-reduce for TP=2 that bypasses NCCL for small (decode-sized) tensors. Each rank
  writes its chunk into pinned host memory mapped into both GPUs (e.g. a /dev/shm segment registered
  with cudaHostRegister mapped), signals with a flag, and reads the peer's chunk; sum in fp32, cast to
  bf16. NCCL stays for large tensors (cutoff ~2 MiB). Bitwise equal to NCCL.
- recognize: new all-reduce backend/kernel in `vllm/distributed/device_communicators/` dispatched
  before PyNCCL for the TP group; host-pinned/zero-copy memory; system-scope fences and flag
  polling. Not this: NCCL env tuning (`nccl-env`); vLLM's stock custom all-reduce (needs P2P,
  disabled here, F1).
- results:
  - on BASE: +5.0% official (`v1/test-3:T1-002`); +5.7% official (`v1/test-4:T1-001`). Decode AR
    61 → 45-50 us, ~1.5-2.2 ms/decode step (`v1/test-3:KB2`, `v1/test-4:KB2`).
  - on BASE + draft-head-diet + sharded-sampling: +5.7% official vs that stack (691.3 vs 653.9,
    `v1/test-4:T1-003`, KB7): stacks nearly additively.
- facts: F1, F3.

### `prefill-graphs`: CUDA graphs for prefill-containing steps
- builds_on: host-ar
- status: win on host-ar; no gain on BASE
- what: serve flag `--compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'` (captures
  mixed prefill/decode steps up to 2048 tokens).
- recognize: that flag, or an equivalent change that graphs the mixed prefill steps. On BASE only
  steps up to 96 tokens are graphed, so prefill steps run eager (`v1/test-4:local_kb/leader-1.md:22`).
- results:
  - on BASE (NCCL): +2.2% `ab`, noise (`v1/test-3:T1-001`, KB1).
  - on host-ar (+ host-allgather): +4.8% `ab` (`v1/test-3:KB4`); official 699.2 vs 662.8
    (`v1/test-3:T1-005`). Super-additive; cause not separated (AR skew vs launch count). Host-staged
    comms also shrink graph memory (`v1/test-3:KB7`).
- grading: a run that tried it only on BASE and dropped it saw a true "no gain on BASE"; the miss is
  not retrying it on host-ar. A run that notes "prefill steps are eager" without testing it came close.

### `host-allgather`: host-staged all-gather for the logits
- builds_on: host-ar
- status: no gain
- results: on host-ar: ~0 `ab` (`v1/test-3:T1-004`, KB6). Kept in test-3's stack as a neutral, exact
  building block (not a contradiction).
- note: `draft-head-diet` and `greedy-fastpath` / `sharded-sampling` remove most logits all-gathers.

### `ar-kernel-tuning`: micro-tuning the host all-reduce kernel
- builds_on: host-ar
- status: no gain
- results: on host-ar: chunk pipelining loses (`v1/test-3:KB5`); unified blocks / per-warp flags
  null `component` (`v1/test-4:T1-002`).
- cause: host-ar is within ~5-10 us of the PCIe floor (F3).

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

### `draft-head-diet`: drafter output-head diet
- builds_on: BASE
- status: win
- what: in the MTP speculative config, `"use_local_argmax_reduction": true` (drafts take the argmax
  per vocab shard and reduce (value, index) instead of all-gathering logits) and
  `"draft_lm_head_quantization": "fp8"` (FP8 W8A8 copy of lm_head for the drafter only; the target
  keeps bf16, so outputs stay exact). Acceptance unchanged (~2.34-2.35).
- recognize: those spec-config keys, or code doing the same.
- results:
  - on BASE: +2.4% `ab` (`v1/test-4:T2-001`).
  - on host-ar + prefill-graphs: +3.0% `ab` (`v1/test-3:T1-007`, KB10); local argmax alone ~+1%.

### `reduced-vocab-draft-head`: draft head over a reduced vocabulary
- builds_on: draft-head-diet
- status: ruled out by estimate
- results: on draft-head-diet: est. net ≈ 0 once the draft head is FP8 (`v1/test-3` idea R1-J,
  ideas.md:43). A measurement overrides this.

### `greedy-fastpath`: exact greedy argmax over vocab shards for verification
- builds_on: BASE
- alternative_to: sharded-sampling
- status: win
- what: per-shard (max, argmax), small all-gather of [rows, 2], cross-rank tie-break so the result
  equals the full argmax exactly (bf16 ties occur in ~1.8% of rows, F10); the rejection sampler
  compares draft ids to target argmax ids. Only for pure greedy requests (temperature 0, no logits
  processors/logprobs).
- recognize: new greedy argmax path in LogitsProcessor / Sampler / RejectionSampler.
- results:
  - on host-ar + prefill-graphs + draft-head-diet: official 730.9, +4.5% vs host-ar + prefill-graphs
    (includes draft-head-diet's ~+3%), so ~+1.5-2% (`v1/test-3:T1-009`, KB10).
  - on BASE + draft-head-diet + sharded-sampling: no gain over sharded-sampling (`v1/test-4:T2-004`).
  - vs sharded-sampling on host-ar + draft-head-diet: 699.5 vs 691.3 official, but ABAB +0.2%: a tie
    (`v1/test-4:T2-006`, journal.md:14, KB5).

### `sharded-sampling`: `--enable-batch-sharded-sampling`
- builds_on: BASE
- alternative_to: greedy-fastpath
- status: win
- what: serve flag; exact, works with MTP on the V2 runner.
- results: on BASE: draft-head-diet + sharded-sampling = +2.8% official (`v1/test-4:T2-003`). Costs
  an NCCL SendRecv of ~0.41 ms/decode step (F11).

### `gdn-replay`: GDN speculative-state replay
- builds_on: BASE
- status: win
- what: stock GDN MTP decode writes k+1 full fp32 recurrent states per step (one per draft position);
  replay writes only the committed state plus a small log of the new tokens' inputs, and replays the
  accepted tokens next step with the same fp32 ops in the same order (bitwise equal). A flush
  materializes the stock layout before anything else reads the state.
- recognize: custom GDN decode kernel in `csrc/` that stores 1 state per step + a token log, with a
  flush/materialize path; hooks in the GDN layer / mamba hybrid code. Not this: GDN backend flags
  (`stock-flag-tweaks`).
- results: on host-ar + prefill-graphs + host-allgather + draft-head-diet + greedy-fastpath: +2.6%
  official, 749.7 (`v1/test-3:T1-012`); same-job A/B +2.2% (712.4 vs 697.3, TPOT p50 19.57 vs 20.41
  ms); ~2 ms/decode step incl. the L2 spill into the next GEMMs (KB15). Expected to be roughly
  independent of that stack; unmeasured on BASE.
- grading: not promoted in test-3 only because of that run's 3% promotion bar.

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

### `draft-depth-3`: 3 (or more) MTP draft tokens
- builds_on: BASE
- status: harmful
- results:
  - on BASE + draft-head-diet: -3.9% `ab`, TTFT p99 2225 ms, gate fails (`v1/test-4:T2-002`).
  - on host-ar + prefill-graphs + draft-head-diet: -6.9% `ab`, TTFT p50 210 → 792 ms, gate fails
    (`v1/test-3:T1-008`, KB11).
  - on host-ar + draft-head-diet + sharded-sampling: flat, TTFT p99 ~2.2 s, gate fails
    (`v1/test-4:T2-005`, KB13).
  - on qwen38_opt's BASE: 2 tokens beat 3 by 12% at c=16 (`qwen38_opt`).
  - with more KV (5.5-6.0 GiB): OOM in graph capture or at runtime (`v1/test-3:T1-010`, T1-011; F6).
  - 4 tokens: the server doesn't start; Marlin cliff at M=80 (`v1/test-3:KB14`).
- cause: acceptance rises (2.35 → 2.72-2.76), but each request holds more GDN spec-state blocks
  (10 → 13), so only 14 of 16 requests fit in the KV pool: admission stalls (F5; `v1/test-3:KB13`,
  KB17). Symptom: TPOT -16% with total duration unchanged, TTFT p50 334 → 938 ms
  (`v1/test-4:agents/leader-2.w2/REPORT_job1.md:30`).
- revisit: via `draft-depth-3-small-state`. Raising `--kv-cache-memory` is not a fresh idea.
- grading: a correct drop cites capacity/admission (F5). "Slower compute", "step cost" or "queueing
  behind prefill steps" without the capacity cause is wrong or incomplete.

### `nccl-env`: NCCL env knobs (protocol, channels, buffer size)
- builds_on: BASE
- status: no gain
- results: on BASE: no gain at decode or prefill sizes `component` (`v1/test-4:KB1`; `v1/test-3`
  idea R1-C).

### `prefill-ar-overlap`: overlap prefill-step all-reduces with compute
- builds_on: BASE
- status: no gain measurable
- results: on host-ar + draft-head-diet + sharded-sampling: works mechanically (Marlin must reserve
  SMs, F9); big prefill steps -10% `component`, end-to-end 0.0% ± 0.7% `ab` (`v1/test-4:T1-004`, KB14,
  KB15). Test-3 traced the prefill AR skew to GPU 0's clock (F2) and set the idea aside
  (`v1/test-3:KB8`, idea R1-L).
- revisit: a more prefill-heavy workload, or finer measurement.

### `triton-linear`: `--linear-backend triton` (true FP8 tensor-core GEMMs)
- builds_on: BASE
- status: harmful
- results: on qwen38_opt's BASE: TTFT -14% but decode ITL +14%, net -4% (`qwen38_opt`).
- revisit: a mixed policy (Triton for prefill, Marlin for decode) is untested.

### `stock-flag-tweaks`: common serve-flag tweaks
- builds_on: BASE
- status: no gain
- results: on qwen38_opt's BASE, all within noise: prefix caching off, Triton attention backend, GDN
  backend flags (flashinfer falls back on sm_89), `--max-num-batched-tokens 4096` (TTFT worse).
  `--no-async-scheduling` is -10% (async is the default) (`qwen38_opt`).

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

## Facts (cited by nodes; also used to judge whether a run's reasons were right)
- **F1.** No GPU P2P: vLLM's custom all-reduce is disabled; TP all-reduce is NCCL over PCIe/host.
- **F2.** Box state (2026-11): GPU 0's SM clock is locked at 2235 MHz, GPU 1 boosts to ~2400-2670
  MHz. Compute-bound kernels ~15-17% slower on GPU 0; this causes the prefill-step AR skew
  (`v1/test-3:KB8`). May change; check the run's gpu_clocks.csv.
- **F3.** PCIe from SMs is not full-duplex here; a T=48 AR has a floor of ~34 us (`v1/test-3:KB5`,
  `v1/test-4:KB3`).
- **F4.** Decode step anatomy on BASE (~35-40 ms): Marlin ~18 ms (~50%), AR 6-8 ms, lm_heads ~4.3
  ms, logits all-gathers ~1.7 ms, GDN decode ~2.4-3.7 ms (+ ~1.7 ms L2 spill). Prefill-containing
  steps ~36% of wall time (`v1/test-4:KB10`, `v1/test-3:KB9`, KB14).
- **F5.** GDN spec-state blocks per request scale with the number of draft tokens: 10 at k=2, 13 at
  k=3, shared with the KV pool (`v1/test-3:KB13`, KB17; `qwen38_opt`).
- **F6.** Memory ceiling: KV above ~5.0 GiB/rank OOMs (graph capture or runtime); graph trimming
  buys ~0.26 GiB (`v1/test-3:KB16`).
- **F7.** FP8 weights run through Marlin (weight-only, bf16 compute) on sm_89; the GEMMs are DRAM
  bound at ~80% of peak (`qwen38_opt` ncu).
- **F8.** Measurement: one server per arm drifts ~2%; close calls need interleaved fresh-server
  repeats (`v1/test-4:KB11`). Short runs can't judge p99 latency (`v1/test-4:KB9`). A ~10% speedup of
  big prefill steps is invisible at c=16 (`v1/test-4:KB15`). Node-level nsys
  (`--cuda-graph-trace=node`) inflates cudaGraphLaunch to ~1.2 ms vs ~120 us at graph level
  (`v1/test-4:KB12`).
- **F9.** Marlin's persistent grid occupies all SMs and blocks concurrent kernels (NCCL included)
  until it drains (`v1/test-4:KB14`).
- **F10.** Greedy verification has bf16 ties for the max in ~1.8% of rows: exact argmax must break
  ties like the stock path (`v1/test-3:KB12`).
- **F11.** `--enable-batch-sharded-sampling` adds an NCCL SendRecv of ~0.41 ms per decode step
  (`v1/test-4:agents/leader-2.w2/REPORT_job1.md:13`).

## History
- 2026-11-07: created from v1 test-3, v1 test-4 and qwen38_opt (test-1/test-2 had no KB).
- 2026-11-07: evidence from the first judged evaluations (evals/v1/test-3, evals/v1/test-4) added;
  all citations checked.
- 2026-11-07: converted to a DAG: status-neutral node ids, explicit `builds_on` / `alternative_to`
  edges, results per base, outline generated by keygraph.py. Old ids: K1 host-ar, K2 prefill-graphs,
  K3 draft-head-diet, K4a greedy-fastpath, K4b sharded-sampling, K5 gdn-replay, D1 draft-depth-3,
  D2 nccl-env, D3 ar-kernel-tuning, D4 prefill-ar-overlap, D5 host-allgather, D6 triton-linear,
  D7 stock-flag-tweaks, D8 pp2 / uneven-tp, D9 dequant-cublas-prefill / reduced-vocab-draft-head,
  O1 spec-glue-graph, O2 small-spec-state, O3 ar-norm-fusion, O4 l2-prefetch, O5 ar-compression,
  O6 ce-ar-prefill, O7 flashinfer-pcie-ar.
