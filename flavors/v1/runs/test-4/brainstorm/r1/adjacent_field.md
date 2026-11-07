# Round 1 ideas: bs-r1-adjacent_field (persona: HPC / interconnect person)

Scripts and raw outputs: `agents/bs-r1-adjacent_field/` (`ar_bench.py`, `pcie_bench.py`, `marlin_l2.py`,
`job.out`, `job2.out`; GPU jobs 1 and 3 in `gpu/jobs/`).

## What the data says (shared evidence for all ideas)
- **Step anatomy, measured** (nsys `task/baseline/nsys_c16/profile.sqlite`, GPU 0, a steady decode step at
  t=135.8–175.5 ms, 39.7 ms span, 38.9 ms busy):
  Marlin 18.6 ms (264 launches) | **NCCL all-reduce 8.1 ms (135 launches, ~60 µs each)** |
  GDN decode 3.7 ms | lm_head GEMMs 4.3 ms (target 1.59 ms at M=48 + drafter 2×1.37 ms at M=16) |
  **logits all-gather 1.67 ms (target 0.96 + drafter 2×0.33)** | rest ~1.5 ms.
- **Whole profile, GPU 0, share of wall time** (measured, same SQLite): Marlin 49.2%, all-reduce 23.4%
  (21.3% the 544-thread T≥48 variant + 2.1% small), lm_head 7.7%, logits all-gather 2.4%, other 12.6%;
  GPU busy 95% (so ~5% host gaps). 550 verify steps in 27.2 s.
- Per layer the pattern is strictly serial: GEMM → AR → norm → GEMM, no overlap.
- **Link, measured** (job 1, `pcie_bench.py`): both GPUs PCIe Gen4 x16, no P2P. SM zero-copy write of
  491 KB (48 tok × 5120 × bf16) to pinned host memory: 19.7 µs (25 GB/s); SM read from host: 21.0 µs
  (23.4 GB/s). Same for copy engines.
- **NCCL, measured** (job 1, `ar_bench.py`, 2 ranks, 128 ARs captured in a CUDA graph like vLLM):
  default (RING_LL, 2 ch): **T=16: 41.6 µs, T=48: 65.5 µs, T=96: 101.7 µs.** Every knob is worse or equal:
  Simple 52/68/104, LL+4ch 41/79/140, LL+8ch 43/82/137, Simple+4ch 49/70/107, Simple+8ch 54/79/117,
  LL128 56/99/157. → **NCCL env tuning is a dead end; don't spend a round on it.**
  Interpretation: NCCL pays ~35 µs fixed cost + moves bytes at ~half the link rate (LL flags).
  Moving the bytes once costs ~20 µs at T=48 and ~7 µs at T=16.
- vLLM's fast paths are all off: `custom_all_reduce` needs P2P (log: "disabled because your platform lacks
  GPU P2P"), FlashInfer PCIe-IPC AR (opt-in `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`) is built on CUDA-IPC
  device buffers (`flashinfer/comm/pcie_ipc_ar.py` → `cuda_ipc.create_shared_buffer`), i.e. peer device
  memory access, which needs P2P. Only PYNCCL is active (`server.log`).

---

## Idea 1 (my main bet): host-staged one-shot all-reduce over pinned shared host memory
- **direction**: Write a 2-rank all-reduce that never needs P2P: both processes map the same
  `/dev/shm` segment and `cudaHostRegister(..., Mapped|Portable)` it, so each GPU gets a device pointer to the
  same host pages. One kernel per AR: each block writes its slice of the local partial into its own
  half of the host buffer, `__threadfence_system()`, sets a per-slice flag (epoch counter, so it is
  graph-replayable); then polls the peer's flag for the same slice, reads the peer slice and adds it
  locally (fp32 add, round to bf16 — same as NCCL's bf16 sum for 2 ranks, so very likely **bitwise
  identical**). Register it as a vLLM all-reduce backend in front of PYNCCL (like `CUSTOM`/`QUICK_REDUCE` in
  `cuda_communicator.py`) for small sizes (≤ ~1–2 MB, i.e. decode/verify/draft); keep NCCL for prefill.
  The same building block also serves the logits all-gather (Idea 2) if wanted.
- **why**: measured above: NCCL takes 65.5 µs (bench) / ~60 µs (server) at T=48 and 41.6/~33 µs at T=16,
  while the link moves those bytes in ~20 µs / ~7 µs. Writes and reads use opposite PCIe directions of
  each GPU's link, so they pipeline across slices. HPC analog: intra-node shared-memory MPI collectives
  ("one-shot / eager protocol for small messages") beat generic ring protocols at this message size.
- **bottleneck**: all-reduce = 23.4% of GPU-0 wall time (measured), 8.1 ms of a 39.7 ms decode step.
- **upside**: target per AR ~25–30 µs at T=48 (assumed: 20 µs transfer + ~5–8 µs flag latency/sync) and
  ~10–12 µs at T=16. Per step: 128 × ~32 µs + ~7 × ~20 µs ≈ 4.2 ms saved of 39.7 → **~+10% tok/s**
  (range +6% if it only reaches 40 µs, +13% if 25 µs). Confidence: medium-high that it beats NCCL clearly;
  the exact floor depends on host-flag polling latency (unmeasured, assumed 1–3 µs per round trip).
  Also lowers TPOT p99 (helps the latency gates).
- **risks**: correctness of the flag protocol (memory ordering across PCIe: use system-scope fences and
  volatile/`ld.acquire.sys` polling); CUDA-graph capture (pure kernel with fixed pointers + device epoch
  counter, so fine); a hung rank spins forever (add a generous spin timeout → trap). No math change.
  Rules: comms stay bf16, nothing lossy → allowed by task_definition.
- **novelty**: nothing tried yet (TREE empty). Different from NCCL tuning (which I measured to be useless)
  and from vLLM's existing custom AR (P2P-only).
- **cheapest test** (~1 h code, 3 GPU-min): standalone 2-process microbench of the kernel at T=16/48/96
  inside a CUDA graph vs NCCL (copy my `ar_bench.py` harness), plus a bitwise compare with NCCL output.
  If T=48 < 40 µs, wire it into vLLM and run a same-job A/B bench vs BASE.

## Idea 2: vocab-parallel head diet (stop shipping logits over PCIe)
- **direction**: Remove the full-vocab logits all-gathers and cut drafter lm_head bytes:
  (a) set `use_local_argmax_reduction: true` in `--speculative-config` (existing flag, `vllm/config/speculative.py:540`,
  used by `speculator.py:557` → `LogitsProcessor.get_top_tokens`): drafter gathers (value, idx) pairs, not 248K logits;
  (b) for the target verification, when every request in the batch is greedy and wants no logprobs, compute
  per-rank argmax on the local vocab shard and reduce (value, index) pairs (lowest index on ties = same as
  full argmax), else fall back to the current gather (generic, not benchmark-keyed);
  (c) FP8 draft lm_head (explicitly allowed by task_definition) to halve the 1.27 GB/GPU it reads twice per step.
- **why**: measured per step: logits all-gather 0.96 ms (target, M=48, ~12–24 MB) + 2×0.33 ms (drafter);
  drafter lm_head 2×1.37 ms (Kernel2 g=8,970, M=16 — pure weight streaming: 124160×5120 bf16 = 1.27 GB/GPU, ~0.93 TB/s).
- **bottleneck**: logits all-gather 2.4% + drafter lm_head ~4.9% of wall (measured shares above).
- **upside**: (a) ~0.65 ms/step ≈ +1.5%, flag only; (b) ~0.98 ms ≈ +2.5%; (c) ~1.37 ms ≈ +3.5% if acceptance
  holds (FP8 head may flip a few draft argmaxes; verification keeps it exact). Total ~+6–8%. Confidence high for
  (a), medium for (b)/(c). Check (a) is compatible with the MTP path here (`_validate_local_argmax_reduction`).
- **novelty**: not tried; the systematic brainstormer may propose parts of it (it's the obvious read of the profile).
- **cheapest test**: (a) one serve flag, same-job A/B bench (~6 GPU-min). (c) offline: compare drafter argmax with
  bf16 vs FP8 head on a few hundred hidden states (non-benchmark text) before any serving run.

## Idea 3: cheaper drafts → deeper drafts (num_speculative_tokens 3), only after 1+2
- **direction**: Once AR and drafter head are cheap, re-tune `num_speculative_tokens` (3, maybe 4).
- **why**: measured acceptance 0.78 / 0.56 (cumulative), AL 2.345. Verify GEMMs are near their DRAM floor at M=48
  (Marlin L2-hot 43–48 µs vs in-server 53–60 µs for the same shapes, job 3) so M=64 costs little in GEMM time;
  the extra cost is one more drafter pass (today ~1.9 ms: MTP layer + 1.37 ms head + AR/gather), which Ideas 1–2 shrink to ~0.8 ms.
- **upside**: assumed 3rd-position cumulative acceptance ~0.40 → AL 2.74 (+17% tokens/step) against ~+5–6% step time
  → ~+8–10%, but only if the drafter is cheap. Confidence low-medium (acceptance at position 3 unmeasured;
  memory: draft count affects KV/graph → 128K test needed).
- **cheapest test**: one flag change, unofficial bench + acceptance metrics, same job as BASE.

## Idea 4 (honest negative, so nobody spends a round on it): L2 prefetch of the next GEMM's weights during the AR
- My original bold idea: DRAM sits idle during each ~60 µs PCIe-bound AR; the 4090 has 72 MB L2, so prefetch the
  next layer's weights (16–45 MB) into L2 on a side stream while the AR runs.
- **measured** (job 3, `marlin_l2.py`, M=48, block-128 FP8 Marlin, per-GPU TP=2 shapes): L2-hot Marlin is
  only 43.6 µs (gdn in_proj 42 MB), 47.8 µs (down 45 MB), 40.9 µs (attn qkv 37 MB) vs ~53/56/~50 µs in the server
  (nsys). (My "cold" numbers, 68–80 µs, are pessimistic: the flush leaves dirty lines.) So Marlin at M=48 is close to
  its own compute/latency floor; prefetch would save ~8–10 µs per GEMM ≈ 1.2 ms/step ≈ **+3%**, with real
  complexity (side-stream kernels, co-residency with NCCL). Low priority; if anything, Marlin's in-kernel
  efficiency (≈44 µs floor for 42 MB = ~0.95 TB/s-equivalent) is not the problem.
- Also ruled out by napkin math: **PP=2 instead of TP=2.** Decode is weight-streaming bound; PP makes each GPU stream
  its 32 full layers once per micro-batch, i.e. 2× bytes per step with 2 micro-batches: ~39 ms vs ~31 ms of GEMM+AR
  per step today (assumed from the per-layer numbers above). Loses unless AR costs > ~2× what it does now.

## Which I'd bet on
1. **Idea 1 (host-staged AR)**: biggest measured gap (60 µs → ~25–30 µs on 23% of the time), exact (likely bitwise),
   and nobody else in vLLM covers this no-P2P case. Needs a capable leader (CUDA + memory-ordering), but the first
   test is a cheap standalone microbench.
2. **Idea 2 (head diet)**: low risk, part (a) is a flag; good second team direction, and it sets up Idea 3.
Skip NCCL env tuning (measured: no gain) and L2 prefetch (measured: ~3%).
