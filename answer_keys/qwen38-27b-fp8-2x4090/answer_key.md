# Answer key: qwen38-27b-fp8-2x4090

A map of the search space for this task, as far as we know it: what works, what doesn't (and on
what base, and why), what nobody has measured yet, and the facts that explain them. It is merged and
curated from all runs, any flavor. Not a copy of any run's graph. It is a lower bound, and it evolves:
format and update rules are in `EVAL.md` ("The answer key").

Metric: output tok/s at c=16 (TASK.md). BASE ≈ 631-636 tok/s official (varies ~1% between runs).
Evidence ids: `<flavor>/<run>:<record or KBn>` (e.g. `v1/test-3:KB13`); `qwen38_opt` = the
hand-driven study in `../qwen38_opt/CLAUDE.md` (its BASE differed: 3 draft tokens, no FP8 KV).
Evidence strength, strongest first: `official` (fresh-server trials), `abab` (interleaved same-job
A/B), `ab` (one same-job A/B), `component` (microbench / nsys of one part), `estimate`.

## Frontier (best known)
**K1 + K2 + K3 + K4a + K5 = 749.7 tok/s official, +18.7% vs BASE 631.4** (`v1/test-3:T1-012`).

## Graph (outline; each node once, "+ X" = other prerequisites, "alt" = alternatives)
```
BASE
├─ K1 host-staged all-reduce ........................ win
│  ├─ K2 prefill CUDA graphs ........................ win on K1 (no gain on BASE)
│  ├─ D3 AR kernel micro-tuning ..................... no gain (near PCIe floor)
│  ├─ D5 host-staged logits all-gather .............. no gain
│  ├─ D4 prefill AR/compute overlap ................. no gain measurable at c=16
│  └─ O3 fuse residual-add + RMSNorm into the AR .... open
├─ K3 drafter head diet ............................. win
│  └─ K4 target-logits diet: alt K4a | K4b .......... win (either one; they substitute)
├─ K5 GDN spec-state replay (+ K1..K4 in the measurement, independent in principle) ... win
│  └─ O2 one GDN spec block per request ............. open (would unblock D1)
├─ D1 3 draft tokens ................................ harmful (capacity stall); blocked by F5/F6
├─ D2 NCCL env knobs ................................ no gain
├─ D6 Triton linear backend (true FP8 GEMMs) ........ harmful
├─ D7 stock serve-flag tweaks ....................... no gain
├─ D8 PP=2 / uneven TP split ........................ ruled out by estimate only
├─ O1 one-graph capture of the spec-decode glue ..... open
├─ O4 L2 prefetch of next GEMM weights during the AR  open
├─ O5 lossless AR payload compression ............... open
├─ O6 copy-engine AR for prefill sizes (needs K1) ... open
├─ O7 FlashInfer PCIe all-reduce flag ............... open (compare with K1)
└─ D9 other estimate-only rule-outs
```

## Wins

### K1. Host-staged 2-rank all-reduce (no P2P)
- **What:** a custom all-reduce for TP=2 that bypasses NCCL for small (decode-sized) tensors. Each
  rank writes its chunk into pinned host memory mapped into both GPUs (e.g. a /dev/shm segment
  registered with cudaHostRegister mapped), signals with a flag, and reads the peer's chunk; sum in
  fp32, cast to bf16. NCCL stays for large tensors (cutoff ~2 MiB). Bitwise equal to NCCL.
- **Recognize:** new all-reduce backend/kernel in `vllm/distributed/device_communicators/`
  dispatched before PyNCCL for the TP group; host-pinned/zero-copy memory; system-scope fences and
  flag polling. Not K1: NCCL env tuning (D2); vLLM's stock custom all-reduce (needs P2P, disabled
  here).
- **Gain:** +5.0% official on BASE (`v1/test-3:T1-002`); +5.7% official (`v1/test-4:T1-001`).
  Decode AR 61 → 45-50 us, ~1.5-2.2 ms/decode step (`v1/test-3:KB2`, `v1/test-4:KB2`).
- **Facts:** F1, F3.

### K2. CUDA graphs for prefill-containing steps (needs K1)
- **What:** serve flag `--compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'`
  (captures mixed prefill/decode steps up to 2048 tokens in CUDA graphs).
- **Recognize:** that flag (or an equivalent change that graphs the mixed prefill steps). On BASE
  only steps up to 96 tokens are graphed, so prefill steps run eager (`v1/test-4:local_kb/leader-1.md:22`).
  A run that notes "prefill steps are eager" without testing this came close.
- **Gain:** on BASE (NCCL) +2.2% `ab`, noise (`v1/test-3:T1-001`, KB1). On K1: +4.8% `ab`
  (`v1/test-3:KB4`); official 699.2 vs 662.8 for K1 (`v1/test-3:T1-005`). Super-additive with K1;
  cause not separated (AR skew vs launch count). Host-staged comms also shrink graph memory (KB7).
- **Grading note:** a run that tried it only on BASE and dropped it saw a true "no gain on BASE".
  Missing the retry on K1 is the miss.

### K3. Drafter head diet
- **What:** in the MTP speculative config, `"use_local_argmax_reduction": true` (drafts take the
  argmax per vocab shard and reduce (value, index) instead of all-gathering logits) and
  `"draft_lm_head_quantization": "fp8"` (FP8 W8A8 copy of lm_head for the drafter only; the target
  keeps bf16, so outputs stay exact). Acceptance unchanged (~2.34-2.35).
- **Recognize:** those spec-config keys, or code doing the same.
- **Gain:** +2.4% `ab` on BASE (`v1/test-4:T2-001`); +3.0% `ab` on K1+K2 (`v1/test-3:T1-007`,
  KB10); local argmax alone ~+1%.

### K4. Target-logits diet (alternatives; reaching either counts)
Avoid all-gathering the target's full logits for greedy verification. K4a and K4b give the same
gain and don't add up (`v1/test-4:KB5`: custom greedy vs sharded sampling, ABAB +0.2%).
- **K4a. Greedy local-argmax fast path (code):** per-shard (max, argmax), small all-gather of
  [rows, 2], cross-rank tie-break so the result equals the full argmax exactly (bf16 ties occur in
  ~1.8% of rows, F10); rejection sampler compares draft ids to target argmax ids. Only for pure
  greedy requests (temperature 0, no logits processors/logprobs). `v1/test-3:T1-009`: official
  730.9, +4.5% vs K1+K2 official (includes K3's ~+3%), so ~+1.5-2% (KB10).
- **K4b. `--enable-batch-sharded-sampling` (flag):** exact, works with MTP on the V2 runner.
  `v1/test-4:T2-003`: K3 + K4b = +2.8% official on BASE.
- On K1+K3 (`v1/test-4`): K4a 699.5 (T2-006) vs K4b 691.3 (T1-003) official, but interleaved ABAB
  +0.2% (`v1/test-4:journal.md:14`): a tie. K4b adds an NCCL SendRecv of ~0.41 ms/decode step (F11).
- **Recognize:** K4a = new greedy argmax path in LogitsProcessor/Sampler/RejectionSampler; K4b =
  the flag.

### K5. GDN speculative-state replay
- **What:** stock GDN MTP decode writes k+1 full fp32 recurrent states per step (one per draft
  position); replay writes only the committed state plus a small log of the new tokens' inputs, and
  replays the accepted tokens next step with the same fp32 ops in the same order (bitwise equal). A
  flush materializes the stock layout before anything else reads the state.
- **Recognize:** custom GDN decode kernel in `csrc/` that stores 1 state per step + a token log,
  with a flush/materialize path; hooks in the GDN layer / mamba hybrid code. Not K5: GDN backend
  flags (D7).
- **Gain:** +2.6% official on K1+K2+K3+K4a (`v1/test-3:T1-012`), ~2 ms/decode step including the
  L2 spill into the next GEMMs (KB15); same-job A/B +2.2% (712.4 vs 697.3, TPOT p50 19.57 vs
  20.41 ms). Not promoted in test-3 only because of that run's 3% bar. Measured only on that base; expected to be roughly
  independent of K1-K4.

## Dead ends (each holds on its base; a stronger result on another base can overturn it)

### D1. 3 (or more) draft tokens: harmful
- Acceptance rises (2.35 → 2.72-2.76), but each request holds more GDN spec-state blocks (10 → 13),
  so only 14 of 16 requests fit in the KV pool: admission stalls, TTFT p99 fails the gate, tok/s
  -3.9% to -6.9% or flat. More KV doesn't fit (F6). 4 tokens: server doesn't start; Marlin cliff at
  M=80 (`v1/test-3:KB14`).
- Evidence: `v1/test-3:T1-008` (-6.9%), KB11, KB13 (cause), KB17; `v1/test-4:T2-002` (-3.9%),
  T2-005 (flat on the champion, KB13); `qwen38_opt` (k=2 beat k=3 by 12% at c=16).
- "More KV" doesn't work: k=3 with 5.5-6.0 GiB KV OOMs in graph capture or at runtime
  (`v1/test-3:T1-010`, T1-011; F6). Raising `--kv-cache-memory` is not a fresh idea.
- Symptoms (`v1/test-4:agents/leader-2.w2/REPORT_job1.md:30`): TPOT -16% but total duration
  unchanged, TTFT p50 334 → 938 ms: the decode gain moves into queueing.
- **Revisit when:** per-request spec state shrinks (O2).
- A correct drop cites capacity/admission (F5); "slower compute", "step cost" or "queueing behind
  prefill steps" without the capacity cause is wrong or incomplete. Test-3 found the cause (KB13);
  test-4 dropped it with the cause "NOT established" (`v1/test-4:KB13`, REPORT_job1.md:30).

### D2. NCCL env knobs (protocol, channels, buffer size): no gain
- `v1/test-4:KB1` (decode and prefill sizes); `v1/test-3` idea R1-C.

### D3. All-reduce kernel micro-tuning beyond K1: no gain (needs K1)
- K1 is within ~5-10 us of the PCIe floor (F3). Chunk pipelining loses (`v1/test-3:KB5`);
  unified blocks / per-warp flags null (`v1/test-4:T1-002`).

### D4. Overlap prefill-step all-reduces with compute: no gain measurable at c=16
- Works mechanically: Marlin must reserve SMs (F9); big prefill steps -10% (nsys). End-to-end 0.0%
  ± 0.7% (`v1/test-4:T1-004`, KB14, KB15). In test-3 the prefill AR skew was traced to GPU 0's
  clock (F2) and the idea set aside (`v1/test-3:KB8`, idea R1-L).
- **Revisit when:** the workload is more prefill-heavy, or measurement gets finer.

### D5. Host-staged all-gather for the logits: no gain (needs K1)
- ~0 (`v1/test-3:T1-004`, KB6). Superseded by K3/K4, which remove most logits all-gathers.
  Test-3 kept it in the stack as a neutral, exact building block: not a contradiction.

### D6. `--linear-backend triton` (true FP8 tensor-core GEMMs): harmful
- TTFT -14% but decode ITL +14%, net -4% (`qwen38_opt`, old BASE). A mixed prefill/decode GEMM
  policy is untested.

### D7. Stock serve-flag tweaks: no gain
- Within noise: prefix caching off, Triton attention backend, GDN backend flags (flashinfer falls
  back on sm_89), `--max-num-batched-tokens 4096` (TTFT worse). `--no-async-scheduling` is -10%
  (async is the default). (`qwen38_opt`, old BASE)

### D8. PP=2, uneven TP split: ruled out by estimate only
- PP=2 doubles weight streaming per token (`v1/test-4:KB10`, napkin). Uneven TP to offset GPU 0's
  clock nets ~0-1.5% by napkin math (`v1/test-3:KB8`, brainstorm r2). Weak evidence: a measured
  result overrides it.

### D9. Other ideas ruled out by estimate only (weak evidence; a measurement overrides)
- Dequantize + cuBLAS for large-M (prefill) GEMMs: ≤ 2% overall
  (`v1/test-3:brainstorm/r2/adjacent_field.md:75`).
- Reduced-vocab draft lm_head: net ≈ 0 once the draft head is FP8 (`v1/test-3` idea R1-J, ideas.md:43).

## Open (no measured result yet)
- **O1. One-graph capture of the spec-decode glue** (drafter passes, sampling, attention-metadata
  builds): ~1.6 ms/step of CPU idle on `v1/test-4:T1-003`, est. +3-4% (`v1/test-4:KB12`).
  Sub-options: skip no-op eager launches in decode (`_zero_kv_blocks`, `_apply_write` x5, mamba
  pre/post), est. 0.5-1.0 ms (`v1/test-4:agents/leader-2.w2/REPORT_job1.md:32`); pure launch fusion
  bounded at 0.1-0.2 ms (`v1/test-4:T2-007`); in-place FlashInfer draft-metadata update, ~1 ms
  (`v1/test-3` idea R1-H). Caution: part of the decode step-start gap seen in node-level nsys is an
  artifact (F8).
- **O2. One GDN spec block per request** (only the replay log slot; needs K5): k=3 could fit 16
  requests in 5 GiB, which unblocks D1. Est. +2-5% (`v1/test-3` ideas R3-A/R3-B v2, KB17).
- **O3. Fuse residual-add + RMSNorm into the K1 kernel** (~1.6 ms/step of norms etc.;
  `v1/test-4` final report).
- **O4. L2 prefetch of the next GEMM's weights during the AR window** (needs K1; after K5 to avoid
  the GDN L2 spill). Est. +3-5%, low confidence (`v1/test-3` ideas R2-C/R3-D). Component measurement caps it near
  +3%: L2-hot Marlin at M=48 is 41-48 us vs ~50-56 us in-server (`v1/test-4:brainstorm/r1/adjacent_field.md:90`).
- **O5. Bitwise-lossless AR payload compression** (needs K1). Est. +2-3% (`v1/test-3` idea R3-C).
- **Marlin in-kernel efficiency:** near its floor by component measurements: the M=48 penalty vs
  M=3 is ~1.6 ms/step (`v1/test-3:KB14`) and L2-hot Marlin is close to its own floor
  (`v1/test-4:brainstorm/r1/adjacent_field.md:90`). Small upside; no change tried.
- **O6. Copy-engine full-duplex AR for large M (prefill)** (needs K1). Est. +1.5-3%, but half the
  prefill AR time is GPU 0 skew (F2) (`v1/test-3` idea R2-E). Low.
- **O7. `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`** (FlashInfer's PCIe all-reduce): a 5-minute
  check, never run (`v1/test-3:brainstorm/r1/systematic_engineer.md:99`). Compare with K1.

## Facts (cited by nodes; also used to judge whether a run's reasons were right)
- **F1.** No GPU P2P: vLLM's custom all-reduce is disabled; TP all-reduce is NCCL over PCIe/host.
- **F2.** Box state: GPU 0's SM clock is locked at 2235 MHz, GPU 1 boosts to ~2400-2670 MHz.
  Compute-bound kernels ~15-17% slower on GPU 0; this causes the prefill-step AR skew
  (`v1/test-3:KB8`). May change; check the run's gpu_clocks.csv.
- **F3.** PCIe from SMs is not full-duplex here; a T=48 AR has a floor of ~34 us
  (`v1/test-3:KB5`, `v1/test-4:KB3`).
- **F4.** Decode step anatomy on BASE (~35-40 ms): Marlin ~18 ms (~50%), AR 6-8 ms, lm_heads
  ~4.3 ms, logits all-gathers ~1.7 ms, GDN decode ~2.4-3.7 ms (+ ~1.7 ms L2 spill). Prefill-containing
  steps ~36% of wall time (`v1/test-4:KB10`, `v1/test-3:KB9`, KB14).
- **F5.** GDN spec-state blocks per request scale with the number of draft tokens: 10 at k=2, 13 at
  k=3, shared with the KV pool (`v1/test-3:KB13`, KB17; `qwen38_opt`).
- **F6.** Memory ceiling: KV above ~5.0 GiB/rank OOMs (graph capture or runtime); graph trimming
  buys ~0.26 GiB (`v1/test-3:KB16`).
- **F7.** FP8 weights run through Marlin (weight-only, bf16 compute) on sm_89; the GEMMs are DRAM
  bound at ~80% of peak (`qwen38_opt` ncu).
- **F8.** Measurement: one server per arm drifts ~2%; close calls need interleaved fresh-server
  repeats (`v1/test-4:KB11`). Short runs can't judge p99 latency (`v1/test-4:KB9`). A ~10% speedup
  of big prefill steps is invisible at c=16 (`v1/test-4:KB15`). Node-level nsys
  (`--cuda-graph-trace=node`) inflates cudaGraphLaunch to ~1.2 ms vs ~120 us at graph level (`v1/test-4:KB12`).
- **F9.** Marlin's persistent grid occupies all SMs and blocks concurrent kernels (NCCL included)
  until it drains (`v1/test-4:KB14`).
- **F10.** Greedy verification has bf16 ties for the max in ~1.8% of rows: exact argmax must break
  ties like the stock path (`v1/test-3:KB12`).

- **F11.** `--enable-batch-sharded-sampling` adds an NCCL SendRecv of ~0.41 ms per decode step
  (`v1/test-4:agents/leader-2.w2/REPORT_job1.md:13`).

## History
- 2026-11-07: created from v1 test-3, v1 test-4 and qwen38_opt (test-1/test-2 had no KB).
- 2026-11-07: first judged evaluations (evals/v1/test-3, evals/v1/test-4, judge a each): added
  evidence to K2, K4, K5, D1, D5, O1, O4, Marlin, F8; new D9, O6, O7, F11. All citations checked.
