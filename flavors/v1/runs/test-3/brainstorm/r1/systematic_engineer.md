# Round 1 ideas: systematic engineer (bs-r1-systematic_engineer)

All "measured" numbers come from my queries of the raw BASE profiles. The scripts are in
`brainstorm/r1/systematic_engineer_queries/` (run from the run folder): q4 = one full decode
step, q5 = the step tail (lm_head and drafts), q9/q10 = per-step NVTX join (prefill vs decode,
graph vs eager), n1 = ncu Marlin pipe and stall metrics. nsys slows the CPU side, so the
eager-mode costs below are upper bounds (see idea 1).

## Where the time goes (measured, nsys c=16, GPU 0)
The step boundaries come from the NVTX ranges `execute_context_N(tok)_generation_M(tok)` (q9).
- **Decode-only steps:** 500 steps, 17.4 s (65% of wall time). A full step (16 reqs x 3 tokens
  = 48 tokens) takes a median of **39.9 ms**. Inside one full step (q4, GPU busy 38.9 of 39.8 ms):

  | part | time | share |
  |---|---|---|
  | Marlin FP8 GEMMs | 18.6 ms (264 launches) | 47% |
  | NCCL AllReduce RING_LL | 8.2 ms (135 calls, ~61 us each, 491 KB) | 21% |
  | lm_head bf16 GEMM | 3 x ~1.37 ms (target + 2 MTP drafts) | 11% |
  | GDN `gdn_decode_post_conv_mtp_kernel` | 3.7 ms (48 x 76 us) | 9% |
  | AllGather | 1.7 ms (2 x 330 us of it gathers the full draft logits) | 4% |
  | norms, attention, small GEMMs, sampling | ~1.5 ms | |
  | idle gaps (CPU: draft metadata rebuilt eagerly, gaps of 110-240 us) | ~1.0 ms | 2.4% |

- **Steps that contain a prefill:** 50 steps, 9.7 s (**35% of wall time**) for 64 prompts.
  - Median step: 147 ms. A fit of step wall time vs prefill tokens gives **99 ms + 0.24 ms per
    prefill token**. A 57-token prefill mixed with 15 decodes takes 128 ms, of which NCCL is 60 ms.
  - **These steps run fully eager**: 1782 cudaLaunchKernel + 706 cuLaunchKernelEx per step, CPU
    range 115 ms, versus 4 ms of CPU for a graphed decode step (q10).
  - Cause: `max_prefill_cudagraph_capture_tokens` defaults to 96 on non-SM100
    (`vllm/config/compilation.py:766`, `vllm/config/vllm.py:~2615`; server log:
    `prefill_cudagraph_capture_token_sizes` up to 96). Any mixed batch over 96 tokens (15 decodes =
    45 tokens + prefill) goes eager.
  - NCCL inside these steps is mostly rank-skew waiting: one rank's CPU launches lag the other's.
- **Cross-check against the official run** (BASE official): 143.5 s for 91.2K output tokens.
  Pure full decode would give 16 x 2.34 / 39.9 ms = **938 tok/s**; official is 631 (67%). Official
  median TTFT is 251 ms, and the first token comes out of the prefill step, so prefill steps cost
  well over 100 ms there too. Estimate: **~35% of the official wall time is in prefill steps**
  (assumed: ~400 prefill steps x ~130 ms).
- **Small batches stay expensive** (q9): batch 1 = 25.7 ms per step, batch 16 = 39.9 ms. The fixed
  cost per step is weight streaming (Marlin 15 ms at M=3) plus all-reduce latency.

## Speed-of-light references
- Weights per GPU are ~12.2 GB FP8 (assumed: computed from config.json shapes; matches the 14.4 GiB
  "model loading" line minus embed and lm_head). At ~0.95 TB/s achievable that is ~12.9 ms. Marlin
  at M=48 takes 18.6 ms (~656 GB/s on average); at M=3 it takes 15.0 ms (~810 GB/s).
- Per-shape bandwidth at M=48 (measured, nsys durations / bytes):
  - gate_up (89 MB): 141 us, 634 GB/s
  - down (44.6 MB): 60 us, 743 GB/s
  - GDN in_proj (42 MB): 56 us, 750 GB/s
  - out_proj / o_proj (15.7 MB): 25-32 us, ~490-630 GB/s
- ncu (n1): tensor pipe 26% active, so the kernel is not compute-bound (the ridge is ~165 flop/B,
  and M=48 is 96 flop/B). Stalls are mostly math_pipe_throttle, then long_scoreboard. Occupancy
  is 1 block per SM, 4-8 warps.
- PCIe: both GPUs are on Gen4 x16, same NUMA node (measured, /sys). AllReduce message size vs
  time: 164 KB takes ~35 us, 491 KB ~61 us. That is a slope of ~12.6 GB/s plus ~22 us fixed. The
  LL protocol carries 50% payload, which matches ~half of PCIe's ~22-25 GB/s. Speed of light for
  491 KB through host memory: ~22 us transfer + a few us of flags = **~27-30 us vs 61 us**.

---

## Idea 1: CUDA-graph the mixed prefill+decode steps (lift the 96-token prefill-graph cap)
- **Direction:** make steps that contain a prefill run under (piecewise/breakable) CUDA graphs
  instead of eagerly.
  - Start by setting `--compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'` (or
    an explicit `prefill_cudagraph_capture_token_sizes` grid up to max_num_batched_tokens = 2048).
    SM100 already defaults to 8192, so the path exists.
  - Then fix whatever breaks: GDN chunked-prefill kernels, FlashInfer prefill planning, MTP drafter
    graphs for mixed batches.
  - Then tune the grid (padding waste vs number of graphs and startup).
- **Why:** prefill steps are 35% of wall time (measured), with a ~99 ms fixed intercept per step
  (measured) against 40 ms for a graphed decode step. They are eager, with ~2500 launches and
  115 ms of CPU per step (measured, q10).
- **Bottleneck:** CPU launch overhead plus rank skew in prefill steps, ~35% of wall time.
- **Upside:** if a prefill step drops from ~130 ms to ~40 ms + 0.15 ms x 262 ≈ 80 ms (assumed),
  that saves ~50 ms x 400 prompts = 20 s of 143.5 s → **+12-20% throughput**. TTFT p99 should
  improve too.
  - Confidence: high that the eager overhead is real and large.
  - Confidence: medium on how much survives without nsys. nsys inflates CPU time, but the official
    median TTFT of 251 ms says these steps are slow in the real run too.
- **Novelty:** untouched (round 1). It is a config plus probable fixes, not a kernel project.
- **Risks:** graph capture memory and startup time grow (needs the 128K-request test, because
  graph sizes change memory). Some mixed-batch path may not be capturable (the breakable graph
  should split around it). Padding up to the next grid size costs compute.
- **Cheapest test:** one GPU job: serve with the flag, check the log for errors and the number of
  captured graphs, run a ~100-prompt c=16 bench, and compare TTFT and throughput with BASE.
  Then one nsys run to confirm that prefill steps are graph launches.

## Idea 2: Faster 2-GPU all-reduce over PCIe without P2P (NCCL protocol/channel tuning → host-memory custom all-reduce)
- **Direction:** all 135 all-reduces per decode step use NCCL RING_LL with 2 channels, even the
  multi-MB prefill ones (measured: kern_sum lists only `AllReduce_Sum_bf16_RING_LL`, grid=2).
  - First, cheap: sweep NCCL_PROTO=Simple (for large messages, or all of them),
    NCCL_MIN/MAX_NCHANNELS, NCCL_SHM_USE_CUDA_MEMCPY / copy-engine options, and NCCL_BUFFSIZE.
    Use a torch.distributed microbench at 16/48/300/2048 tokens x 5120 bf16, inside a CUDA graph.
  - If the gap to ~30 us remains: a custom 2-rank all-reduce through pinned, mapped host memory.
    Each GPU writes its partial to host memory with chunked flags (PCIe posted writes are ordered,
    so a flag written after the data with threadfence_system is safe), reads the peer's chunks,
    and sums locally. For 2 ranks a+b is commutative, so the result is bitwise the same on both
    ranks and the same math as NCCL. Optionally fuse it with the following residual-add + RMSNorm.
  - Check `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` (FlashInfer 0.7.2 ships it) in 5 minutes. It
    is CUDA-IPC based, so I expect it to fail without P2P on 4090 (assumed), but it costs nothing
    to check.
  - Exactness: the protocol only changes transport. Precision stays bf16; no quantized comms.
- **Why:** 8.2 ms of the 39.9 ms decode step (21%), measured. 61 us vs a ~27-30 us speed of light
  for 491 KB (napkin, from the measured 12.6 GB/s slope and Gen4 x16). In prefill steps, comm per
  token is bandwidth-bound at LL's ~50% efficiency.
- **Bottleneck:** all-reduce, 21% of decode steps, and a large part of prefill steps.
- **Upside:** NCCL tuning alone, ~61 → ~42 us: ~2.6 ms/step ≈ **+6% decode**, plus more on
  prefill tokens. A custom host all-reduce at ~30 us: ~4.2 ms/step ≈ **+10-11% decode**. This
  also helps the small-batch tail, where the AR latency floor matters.
  - Confidence: medium (NCCL defaults are sometimes already right; the custom path is real
    engineering, ~1 round).
- **Novelty:** untouched.
- **Cheapest test:** one GPU job, a ~5 min microbench over env combos and sizes. No server needed.
  - Note for the manager: the official eval takes flags only (serve_args.txt). Env defaults would
    have to be set in the diff (e.g. in vLLM's distributed init). The leader should get that
    approved.

## Idea 3: Hide weight streaming under communication: prefetch the next GEMM's weights into L2 during each all-reduce
- **Direction:** during each all-reduce the GPU's DRAM is idle: NCCL uses 2 SMs and spins on host
  flags. On a side stream (forked/joined inside the CUDA graph), launch a read-only prefetch
  kernel that pulls the first ~40-55 MB of the next Marlin GEMM's weights into the 72 MB L2:
  - after the attention/GDN out all-reduce → gate_up (89 MB; prefetch the first ~50 MB);
  - after the MLP all-reduce → the next layer's GDN in_proj (42 MB) or qkv (~37 MB), which fit
    entirely.

  The GEMM then hits L2 for those bytes. This is read-only, with no data hazards, and the math is
  unchanged.
- **Why:** 135 AR windows x ~61 us = 8.2 ms per step, measured, with DRAM idle. 61 us x
  ~0.9 TB/s ≈ 55 MB per window could be prefetched (napkin). Marlin is ~47% of the step and
  streams from DRAM at 634-750 GB/s (measured).
- **Bottleneck:** Marlin DRAM time (47%), using the AR window (21%).
- **Upside:** ~90 MB per layer moved off the critical path ≈ ~90 us per layer saved after
  subtracting the L2-read time, x 64 ≈ **~5-6 ms/step ≈ +12-15% decode** (napkin, assumed L2
  retention of ~70%+).
  - Confidence: low-medium. Uncertain: Ada L2 retention/hashing when 50 MB is prefetched; whether
    Marlin's cp.async evict hints still hit; whether the prefetch kernel slows NCCL's polling.
  - Partly competes with idea 2: a faster AR shrinks the window (at a 30 us AR, roughly half this
    upside). Still additive: idea 2 removes time, idea 3 fills what remains.
- **Novelty:** untouched; not in vLLM.
- **Cheapest test:** one GPU microbench job:
  1. Marlin (M=48, gate_up and in_proj shapes, real weights' layout) cold-L2 vs after a prefetch
     kernel → gives the maximum gain.
  2. NCCL 491 KB all-reduce alone vs concurrent with the prefetch kernel on another stream →
     checks that the AR is not slowed.

  If (1) shows <15 us of gain per GEMM, drop the idea.

## Idea 4: Retune Marlin for the M=48 decode shapes (config choice per shape; small-N GEMMs)
- **Direction:** at M=48 (16 reqs x 3 verify tokens), Marlin bandwidth varies a lot by shape:
  gate_up uses the 256-thread / thread_n=16 / thread_k=4 config and gets 634 GB/s; down and
  in_proj use the 128-thread n=4/k=8 config and get ~745 GB/s; the 15.7 MB out_proj/o_proj get
  ~490 GB/s (measured).
  - First: sweep Marlin's per-shape config selection (thread_k/n, stages, fp32-reduce/atomic
    flags) in a standalone microbench on the 6 real shapes at M=16, 48 and 64. Hardcode the
    winners for sm_89.
  - Further: an sm_89-specific improvement for small-N GEMMs (e.g. split-K so that 128 SMs keep
    enough bytes in flight).
  - Different kernels with the same math are allowed (accumulation order only).
- **Why:** 18.6 ms/step in Marlin at M=48 vs 15.0 ms at M=3 for the same bytes (measured). At
  M=48 it pays ~3.6 ms for compute/dequant overhead without being compute-bound (tensor pipe
  26%, ncu). The speed of light is ~12.9 ms.
- **Bottleneck:** Marlin, 47% of decode steps (and the prefill GEMMs).
- **Upside:**
  - Every shape at the best observed config's ~745 GB/s: gate_up 141 → 120 us and small GEMMs
    +30% → ~2 ms/step ≈ **+5%**.
  - Reaching ~850 GB/s overall: ~4 ms ≈ +10%.
  - Confidence: medium for the config part, low for a new kernel in one round.
- **Novelty:** untouched.
- **Cheapest test:** a GPU-only microbench (no server) on the 6 shapes; ~5 min.

## Idea 5: Trim the MTP drafting overhead (exact parts first)
- **Direction:**
  - (a) Draft logits: each of the 2 draft passes all-gathers the full vocab-parallel logits
    (330 us each, measured) before argmax. Replace that with a local per-shard argmax plus a gather
    of (max, index) pairs. Exact: greedy draft tokens are unchanged, with first-index tie-break.
  - (b) Remove the ~1 ms of CPU gaps per step: FlashInfer "cannot update draft decode metadata in
    place; falling back to rebuilding attention metadata eagerly" (server log), with gaps of
    110-240 us between draft kernels (measured). Implement the in-place update for FlashInfer, or
    use a backend that supports it.
  - (c) **Needs a manager ruling**: a draft-only reduced lm_head (e.g. the top-32K token ids from a
    generic, non-ShareGPT corpus, FR-Spec style) for the 2 draft passes. Verification stays
    full-vocab, so output is exactly the target's; only acceptance can change. Each draft lm_head
    reads the full 1.27 GB shard for 16 rows (1.37 ms, measured).
- **Why / bottleneck:** draft lm_heads 2.7 ms + draft all-gathers 0.66 ms + gaps ~1 ms, out of 39.9 ms.
- **Upside:** (a) ~0.6 ms ≈ +1.5%; (b) up to ~1 ms ≈ +2.5%; (c) ~2.4 ms ≈ +6% if acceptance holds.
  - Confidence: high for (a), medium for (b), unknown legality for (c).
- **Novelty:** untouched.
- **Cheapest test:** (a) and (b) are code changes testable with a short bench; check the
  acceptance numbers are unchanged for (a).

## Idea 6 (lower priority): GDN spec-decode state traffic: write one state instead of three
- **Direction:** the GDN MTP decode kernel (`csrc/gdn/fused_gdn_decode_kernel.cu:~322`) writes the
  full fp32 recurrent state after **every** draft position: 3 x 24 MB written + 24 MB read per
  layer at c=16 (napkin: 24 heads x 128x128 x 4 B x 16 seqs). Instead, keep the committed state
  and the per-token (q, k, v, decay, beta), and replay the accepted tokens at the start of the
  next step (cheap compute). Only one state per step is written. Exact: the same fp32 ops in the
  same order.
- **Why:** 3.7 ms/step (9%), measured. Traffic would drop from ~96 MB to ~48 MB per layer.
- **Upside:** ~1.5-1.8 ms/step ≈ +4%. Confidence: medium on the number, high effort
  (spec-decode state bookkeeping, prefix-cache "align" mode).

## Ruled out (with numbers)
- **PP=2 instead of TP=2:** each GPU reads its full half per micro-batch, so 2 micro-batches
  double the weight traffic. lm_head and MTP pile onto the last stage (+~8 ms per micro-batch).
  Estimated ~58 ms per 16-request round vs 40 ms for TP. TPOT also roughly doubles (gate).
- **3 draft tokens:**
  - Cost: +~2.1 ms per extra draft (MTP layer + 1.37 ms lm_head + gather) plus a larger verify
    (64 tokens, AR +~13 us each): ~+11% step time.
  - Gain: accept length 2.34 → ~2.7 assumed (+16%) → net ~+4%, which is at the noise limit. It
    becomes better after ideas 2 and 5 cut per-token and per-draft costs. A free flag to re-check
    later, not a direction.
- **FP8/W8A8 lm_head or activations:** forbidden by task_definition.

## Combinations
- **Idea 1** (prefill steps, CPU) is independent of everything else: its gain adds to any
  decode-step gain.
- **Ideas 2 and 3** share the all-reduce window: partially overlapping, do idea 2 first.
- **Ideas 3 and 4** both target Marlin: if 3 moves bytes into L2, Marlin's compute/dequant
  efficiency (idea 4) becomes the limit, so they complement rather than duplicate.
- **Idea 5** adds to all. **Idea 6** is independent.

## My bets
1. **Idea 1:** the biggest measured gap (35% of wall time in eager prefill steps), the cheapest
   test (one flag), and high confidence that it matters. If only one direction is picked, this
   one.
2. **Idea 2:** 21% of decode plus a large part of prefill, a clear ~2x gap to speed of light, and
   a 5-minute microbench decides whether the cheap version is enough.
3. **Idea 4 or 3** as the next round's decode-kernel direction, once 1 and 2 have shifted the
   profile. Idea 4 is safer; idea 3 has more upside and less certainty.
