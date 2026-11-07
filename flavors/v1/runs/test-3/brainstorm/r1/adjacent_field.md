# Round 1 ideas — bs-r1-adjacent_field (HPC / distributed-systems view)

## What I measured first (raw data; scripts in /tmp/bs_adj, microbench in brainstorm/r1/adjacent_field_mb/)

**M1. Where the timeline goes** (baseline nsys, GPU 0, 551 engine steps in 27.2 s; steps split at `_prepare_decode_inputs_kernel`).
- GPU 0 is busy 95% of the time (25.9 of 27.2 s). CPU/launch gaps are small, so this is a GPU-time problem.
- **Decode-only steps: 501 steps, 17.4 s (64%), median 38.4 ms.**
  **Prefill-containing steps: 49 steps, 9.6 s (36%), mean 196 ms, median 149 ms.** One prompt of about 260 tokens
  stalls the other 15 sequences for about 150 ms. This explains ITL p99 of 299 ms in bench.log.
- Typical decode step (38.5 ms, measured): Marlin 18.6 ms (48%) · NCCL all-reduce 7.8 ms (20%, 135 calls of
  about 57 µs) · lm_head + logits all-gather 5.6 ms (14.5%; see M3) · GDN decode 3.0 ms (7.7%) · everything else about 3.5 ms.
- Prefill steps (measured, all 49): Marlin about 47% · **NCCL all-reduce 32.9% (135 calls/step, mean 478 µs each)**.
- One GDN layer in decode (measured, about 498 µs): in_proj Marlin 56 µs → small GEMM + conv 14 µs → gdn_decode
  62 µs → out_proj Marlin 33 µs → **AR 59 µs** → norm 4 µs → gate_up Marlin 140 µs → act 3 µs → down Marlin 61 µs →
  **AR 59 µs** → norm 4 µs. The two ARs are 24% of the layer. During an AR, DRAM and 126 of 128 SMs sit idle:
  the NCCL kernel has grid = 2.
- AR across all steps: 6.35 s of GPU 0's 27.2 s (23%). About 1.4 s of that is skew, i.e. waiting for the peer
  (sum of per-AR duration 6.35 s vs. post-arrival duration 4.94 s).

**M2. All-reduce microbenchmark (my GPU job 2: 2 procs, CUDA-graph-captured, bf16 [M,5120]).** Measured:
| config | M=48 (491 KB, decode) | M=450 (4.6 MB, prefill) | M=2048 |
|---|---|---|---|
| NCCL default | 64 µs (7.7 GB/s) | 347 µs (13.3 GB/s) | 1480 µs (14.2) |
| NCCL_PROTO=Simple | 64 µs | 347 µs | 1484 µs |
| NCCL_PROTO=LL128 | 83 µs (worse) | 574 µs (worse) | 2511 µs |
| NCCL_MIN_NCHANNELS=8 | 69 µs (worse) | 326 µs (−6%) | 1370 µs (−7%) |
| 16 channels | 73 µs (worse) | 346 µs | 1347 µs |
- Not measured (the job hit its 12-min timeout, since each torchrun takes about 2 min): Simple+8ch, NCCL_SHM_USE_CUDA_MEMCPY=1.
  A leader could add them to `adjacent_field_mb/job.sh`. I don't expect them to reach the floor.
- Raw PCIe with copy engines, pinned host memory, 4.6 MB: D2H 25.5 GB/s, H2D 24.7 GB/s, **both directions at once
  20.1 GB/s each** on both GPUs. Both GPUs are Gen4 x16 on NUMA 0 (`/sys/bus/pci/.../current_link_speed`).
- **Interpretation:** a 2-GPU all-reduce through host memory has to push S bytes out of and S bytes into each GPU,
  whatever the algorithm (ring and one-shot are the same here). The floor is about S / 20 GB/s plus sync latency:
  - 491 KB → about 25 µs + about 5–10 µs, against NCCL's 57–64 µs: **about 1.7–2x headroom in decode**.
  - 4.6 MB → about 230 µs against 347 µs (microbench) or 478 µs (in the server, larger M): **about 1.5–2x headroom in prefill**.
  - NCCL env-var tuning **does not** get there. The only measured gain was −6% at large M with 8 channels, and it is
    worse at decode size. So a flag-only approach is ruled out, which saves the team a round.
- vLLM's `FLASHINFER_PCIE_IPC` all-reduce backend (opt-in env var) needs cudaIpc **device** memory mapped across
  GPUs (`flashinfer/comm/cuda_ipc.py:268`), i.e. P2P. Expect it to fail or fall back on this box. That is cheap to
  confirm, but don't count on it.

**M3. lm_head and logits** (decode step, measured): the target lm_head (bf16, 1.27 GB per GPU) is a cutlass Kernel2 grid (8,243) at 1.45 ms.
Each of the 2 MTP drafts runs a full-vocab lm_head, Kernel2 (8,970), at 1.36 ms each. The full-logits all-gathers
cost 0.84 ms (target) + 2×0.29 ms (drafts). Total 5.6 ms per step.

## Idea 1 (my top bet) — Host-staged "one-shot" all-reduce built for no-P2P PCIe, using copy engines (CE)
- **direction:** Replace NCCL for the TP all-reduces with a custom AR designed for this topology:
  1. Each rank writes its partial to a slot in a shared pinned host buffer (`/dev/shm` + `cudaHostRegister`;
     NCCL's SHM transport already does this here, so it works without root).
  2. It raises a flag, waits for the peer's flag, reads the peer's partial and adds locally.
  3. Use wide coalesced stores and many blocks (or copy-engine memcpys + `cuStreamWrite/WaitValue32` on host
     flags; both can be captured in CUDA graphs), so both PCIe directions are busy at once.

  Because a+b == b+a exactly, both ranks get bitwise identical bf16 sums, so it stays at baseline precision.
  Register it as one more backend in `cuda_communicator.py`, next to PYNCCL. Also use it for the logits all-gathers.
- **why:** see M1/M2. NCCL achieves 7.7 GB/s (decode) / 13.3 GB/s (prefill), against a measured 20 GB/s
  bidirectional link. NCCL tuning doesn't close the gap (measured).
- **bottleneck:** all-reduce, 23% of total GPU time. That is 20% of decode steps and 33% of prefill steps (measured).
- **upside (napkin):**
  - Decode: 57 → about 35 µs × 135 calls saves about 3 ms of 38.5 ms, i.e. about 8% of decode time.
  - Prefill: 478 → about 250 µs saves about 16% of prefill-step time.
  - Overall about 2.9 s of 27 s → **+8 to +12% tok/s**. TTFT p99 and TPOT p99 should also improve.

  Confidence: medium. The floor is measured. Whether a custom kernel reaches about 80% of it is assumed. Sync
  latency through host memory (about 2–5 µs per flag round trip, assumed) is the main risk at decode size.
- **novelty:** nothing is in BANK. vLLM's own custom AR and the flashinfer PCIe IPC backend both assume P2P. This one doesn't.
- **cheapest test:** a 2-process microbenchmark (extend `brainstorm/r1/adjacent_field_mb/ar_bench.py`, already
  set up). Prototype it in Triton/CUDA against host-mapped pinned memory at M=48 and M=450. Compare with the NCCL
  numbers above, under CUDA graph capture. Takes 1–2 GPU-min. Go/no-go: M=48 ≤ 40 µs.

## Idea 2 (bold) — Use the dead time during each all-reduce to prefetch the next layer's FP8 weights into L2
- **direction:** During each AR (about 59 µs in decode), DRAM and 126/128 SMs are idle (M1). On a forked stream
  inside the CUDA graph, launch a small "prefetch" kernel. It reads the next GEMM's weight bytes with an
  `evict_last` L2 hint (`ld.global.L2::evict_last` / `prefetch.global.L2::evict_last`; both exist on sm_89). Join
  the stream before that GEMM. The 4090 has a 72 MB L2. The next weights fit:
  - after the MLP AR comes in_proj / qkv: 43 MB (GDN) or 37 MB (attention);
  - after the attention AR comes gate_up (91 MB), of which about 45–50 MB can be prefetched.

  Marlin then reads part of its weights from L2 instead of DRAM. This is the classic HPC move of hiding
  communication behind data movement for the next phase. The math is unchanged.
- **why:**
  - Measured: Marlin runs at about 676 GB/s average (196 MB per layer in 290 µs) and is DRAM-bound.
  - Measured: the ncu key csv shows Marlin weight bytes all come from DRAM (43 / 16 / 91 / 46 MB read, i.e. the
    weight sizes).
  - Measured: a 59 µs window × about 0.9 TB/s ≈ 53 MB of prefetch per window.
- **bottleneck:** Marlin, 48% of decode time, plus the idle AR windows (20%).
- **upside (napkin, assumed L2 behaviour):** if in_proj comes fully and gate_up half from L2, saving is about
  40 + 55 µs per layer of 498 µs. That is up to about 15% of decode time as an upper bound; realistically
  **+4 to +8% tok/s**. It shrinks if Idea 1 shortens the AR windows; the two compete for the same 20%.
  Confidence: low–medium. Two unknowns: how much faster Marlin really gets with L2-resident weights (it may become
  latency-bound at only 4 warps/SM, per ncu `warps_active` 8%), and L2 capacity/eviction on Ada.
- **novelty:** high. I haven't seen it in vLLM. It needs no new kernel math, just a side-stream read kernel and a graph fork/join.
- **cheapest test:** a single-GPU microbenchmark, about 1 GPU-min. Run the Marlin FP8 GEMM with in_proj shape
  (M=48, K=5120, N=8192 per rank) in two conditions:
  - (a) after flushing L2 with a 128 MB write;
  - (b) right after a read kernel over its weight buffer.

  If (b) is ≥1.6x faster, then test concurrency with NCCL (prefetch + AR in parallel, check the AR isn't slowed).

## Idea 3 — Hide the prefill all-reduce behind compute with two micro-batches (prefill is compute-bound, so splitting is nearly free)
- **direction:** In prefill steps (36% of time), split the token batch into two halves A and B, and pipeline
  layer by layer: AR(A) runs while GEMM(B) computes, then the reverse. The standard argument against
  micro-batching ("each half re-reads the weights") doesn't apply here, because prefill GEMMs at M ≈ 300–600 on
  bf16-with-fp32-accumulate are compute-bound. The AR should go over copy engines or a low-SM kernel (Idea 1's
  transport), so it doesn't steal SMs from the GEMM. This only reorders work, so the outputs are identical.
- **why:** AR is 32.9% of prefill-step time (measured: 478 µs × 135 per step, about 65 ms of a 196 ms step).
- **bottleneck:** prefill-step AR, about 12% of total GPU time (measured: 0.329 × 9.6 s / 27.2 s).
- **upside:** hiding about 60% of it gives about 7% of total time, i.e. **+5 to +8% tok/s** and a large TTFT/ITL p99 improvement.
  It stacks with Idea 1, with diminishing returns. Confidence: medium-low (the engineering is heavy: vLLM's
  forward isn't written for intra-step micro-batching, and GDN chunked prefill state must be threaded per
  sub-batch). It is better as a round-2 follow-up to Idea 1.
- **novelty:** like NanoFlow/TokenWeave-style overlap; nothing in BANK.
- **cheapest test:** a 2-process microbenchmark: one row-parallel GEMM of [450,8704]×[8704,5120] bf16 plus an AR of
  the output, (a) serial vs. (b) split in 2 with the AR on a second stream. Measure the overlap achieved. About 1 GPU-min.

## Idea 4 — Logits "diet": stop shipping and computing full-vocab logits when an argmax is all that's needed
- **direction:** Three exact pieces, in increasing order of effort and validity risk:
  - **(a) flag:** `--speculative-config '{..., "use_local_argmax_reduction": true}'`. It exists in this tree
    (`vllm/config/speculative.py:540`) and replaces the draft logits all-gather (2×0.29 ms) with a
    (value, index) gather.
  - **(b) the same for the target's verification** when every request in the batch is greedy with no logprobs:
    take the local argmax per vocab shard and gather 8 bytes per token instead of 48×124K logits (0.84 ms).
    This is a generic fast path, not benchmark detection.
  - **(c) cheaper draft lm_head.** The 2 draft heads cost 2.7 ms per step and only change acceptance, never the
    output. Two options: `draft_lm_head_quantization: "fp8"` exists in this tree but **uses FP8 activations: ask
    the manager** whether that is allowed for the drafter only. The alternative is a draft head over a vocab
    prefix: Qwen ids are roughly frequency-ordered, and slicing rows needs no copy. Measured on vLLM docs (600K
    tokens, not benchmark data): ids < 32K cover 90.8%, < 64K cover 96.7%. A 64K prefix would make each draft
    head about 4x cheaper for about 3% fewer accepts.
- **why:** M3, 5.6 ms of the 38.5 ms decode step.
- **upside:** (a) +1.5% · (b) +2% · (c) about +3–5% net. Together **+4 to +7%**. Confidence: high for (a)/(b), medium for (c).
  Synergy: cheaper drafting makes a 3rd MTP token more attractive (per-position acceptance 0.80 / 0.61 measured
  in bench.log).
- **novelty:** low-ish; a systematic engineer will likely see (a). I list it because (a) is nearly free.
- **cheapest test:** (a) is one flag in a short bench. For (b)/(c), first check that greedy outputs are identical on 50 prompts.

## Ruled out (napkin math, to save a round)
- **PP=2 instead of TP=2:** it removes the ARs but doubles weight reads per step. Each GPU must stream its 13.5 GB
  once per micro-batch, so about 37 ms of GEMM vs. today's 18.6 + 7.8 ms AR. That is worse (assumed 2
  micro-batches to keep both GPUs busy).
- **NCCL env tuning (protocol / channels):** measured in M2, at most −6% at prefill size and worse at decode size.

## My bets
1. **Idea 1** (custom host-staged AR). It has the largest measured gap (1.5–2x on 23% of GPU time), it is exact,
   and its first test is cheap and already half-built.
2. **Idea 4a+4b** as a cheap add-on any team can include.
3. **Idea 2** as the bold one. It's a 1-GPU-min microbenchmark to kill or confirm, and if L2 behaves, it is a new lever on the 48% Marlin share.
