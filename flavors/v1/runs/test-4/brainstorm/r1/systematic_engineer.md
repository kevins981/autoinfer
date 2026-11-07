# Round 1 ideas: systematic engineer (bs-r1-systematic_engineer)

## Where the time goes (measured, baseline nsys `task/baseline/nsys_c16/profile.sqlite`, GPU 0)
Scripts: `agents/bs-r1-systematic_engineer/a2.py, a3.py, a4.py` (steps split at `_prepare_pos_seq_lens_kernel`,
one per engine step; a step counts as "prefill" if it holds `chunk_fwd_kernel_o`).

| | steps | mean wall/step | share of wall | AR (all-reduce) per step |
|---|---|---|---|---|
| decode-only steps | 500 | 34.8 ms | 64% (17.4 s) | 6.2 ms |
| steps with a prefill chunk | 50 | 194.7 ms | 36% (9.7 s) | **64.9 ms** |

- All-reduce total = 50x64.9 + 500x6.2 = **6.35 s of 27.1 s wall = 23%**; half of it sits in the prefill steps (measured).
- Median decode step (38.5 ms span), per step (measured, a4.py): Marlin 17.5 ms (264 launches), NCCL AllReduce
  6.2 ms (135 launches, ~46 us avg, 491 KB at 48 tokens), **lm_head GEMMs ~4.2 ms** (cutlass `Kernel2` grid 8x970 =
  vocab/2 rows, 1.36 ms each, ~3 per step: target M=48 + 2 MTP draft steps M=16; ~935 GB/s, already at DRAM roofline),
  **NCCL AllGather of logits 1.1 ms** (5/step; ~330 us clusters = drafter, ~950 us = target), GDN decode 2.4 ms,
  everything else < 2 ms.
- Marlin (ncu `marlin_c16.ncu-rep`): 630-805 GB/s = 64-84% of DRAM peak; top stall `math_pipe_throttle`, 1 block/SM.
  Headroom ~15-20% on Marlin, realistically less.
- AR timing, decode (a3.py): ~57 us after both GPUs entered (491 KB) = 8.6 GB/s algorithm BW. PCIe link is Gen4 x16,
  host-staged copies reach 23-25 GB/s per direction (measured by bs-r1-adjacent_field, gpu job 1). Prefill-step ARs:
  p50 343 us, p90 1.1 ms, and a p50 **start skew of 174 us** between the GPUs (one waits for the other).
- NCCL: `RING_LL` for every size (no P2P -> SHM through host). bs-r1-adjacent_field's bench (gpu job 1): NCCL_PROTO=Simple
  gives no gain at T=16/48/96 (41.8/65.7/103 us LL vs 52/68.9/106.6 us Simple). My own bench for T=400/2048 (gpu job 2) hung and timed out
  with no output (probably the graph-capture of NCCL in my script), so NCCL Simple vs LL at prefill sizes (T~300-2048)
  is still unmeasured: worth 1-2 GPU-min early in idea 2's round, since if Simple is ~2x there it's a free env-var win
  for prefill steps.
- PP=2 instead of TP=2 is ruled out by napkin math: decode is weight-bandwidth-bound, and PP needs 2 microbatches of
  8 seqs to keep both GPUs busy, so each GPU reads its half of the weights twice per 16-seq step: +17.5 ms Marlin
  to save ~6-7 ms of AR. Not proposed.

---

## Idea 1: Drafter head diet (FP8 draft lm_head + local-argmax drafting), then re-tune draft depth
- **direction**: Turn on two existing vLLM speculative-config options: `"draft_lm_head_quantization":"fp8"` (MTP
  supports it: `vllm/v1/worker/gpu/spec_decode/mtp/config.py` inherits Eagle's `supported_options`; Model Runner V2
  only, which the baseline uses: `speculator.py` in the log) and `"use_local_argmax_reduction":true` (Qwen3.5 MTP
  implements `get_top_tokens`, `vllm/models/qwen3_5/nvidia/mtp.py:277`). Both are explicitly allowed by
  task_definition.md (FP8 draft head inside the drafter; the target still verifies with its bf16 head). Then, with
  the cheaper drafter, re-measure `num_speculative_tokens` 2 vs 3.
- **why**: measured: each draft step reads the full bf16 lm_head (1.27 GB/GPU, 1.36 ms) and all-gathers 16x124160
  logits (~330 us). Per decode step the drafter spends ~2x1.36 + 2x0.33 = **3.4 ms of 34.8 ms (~10%)** on its head.
  FP8 halves the head read (-1.36 ms); local argmax removes the gathers (-0.66 ms). Code check: the FP8 copy uses
  per-token activation scales on "the platform's FP8 linear kernel" (sm_89 has cutlass FP8 scaled_mm). Leader should
  confirm both options compose (get_top_tokens through the quantized head) and that the FP8 path doesn't fall back to
  something slow on sm_89.
- **bottleneck**: drafter lm_head + logits gather, ~10% of decode-step time; decode steps are 64% of wall.
- **upside**: flags alone: -2.0 ms/decode step = ~5.8% of decode -> **~+4% total** (assumed: acceptance unchanged within
  ~1%, FP8 drafter head keeps argmax almost always). Draft depth 3: per-position acceptance 0.78/0.56 (measured,
  BASE official); guessing pos-3 ~0.40 (assumed, same 0.72 decay) -> acceptance length 2.35 -> ~2.74 (+17% tokens/step)
  for ~+3.5 ms/step (extra MTP pass ~1 ms with FP8 head, +16 verify tokens: AR bytes +33%, GDN/attention scale, Marlin
  M 48->64 roughly flat). Net estimate **+4-8% more**, low-medium confidence (acceptance at depth 3 unknown; TPOT p99 must
  stay < 55.3 ms). Combined 1+depth: +6-12%.
- **novelty**: nothing tried yet (tree empty). Pure config: serve_args.txt only, no build.
- **cheapest test**: one GPU job, same server binary, three back-to-back benches (100-200 prompts): BASE flags;
  +fp8 head +local argmax; same with num_speculative_tokens=3. Read acceptance length and tok/s. ~15 GPU-min. Memory
  change is small (FP8 head copy +0.64 GB/GPU: run `long_request 128000` once since the KV budget is fixed at 5 GiB
  but activation memory changes).

## Idea 2: Host-staged custom all-reduce for no-P2P PCIe (replace NCCL RING_LL)
- **direction**: Write a CUDA-graph-capturable 2-rank all-reduce that goes through pinned, mapped host memory shared
  by the two worker processes (cudaHostRegister on a shm segment, or NCCL-style), pipelined in chunks with flags so
  that each GPU's PCIe write (its partial) and read (the peer's partial) overlap on the full-duplex link; each GPU sums
  locally (a+b == b+a in bf16, so both ranks get bitwise-identical results; precision unchanged = allowed). Plug it in
  as an all-reduce backend in `cuda_communicator.py` for sizes up to the prefill max, NCCL as fallback. (The existing
  FlashInfer PCIe-IPC backend needs P2P peer access to device memory, which this box lacks: p2p False.)
- **why**: measured: AR = 23% of wall; decode AR ~57 us for 491 KB (8.6 GB/s) while host copies reach 23-25 GB/s per
  direction; "both GPUs write+read 491 KB, no overlap" = 48.6 us (adjacent_field's bench), so a pipelined version
  should land ~25-30 us (assumed: 20 us wire + ~5-10 us flag latency). Prefill-step ARs (3.9 MB-ish) run at ~7-8 GB/s
  under LL (2x wire overhead of LL's flag words) -> ~2.5x headroom. NCCL protocol knobs do not close this
  (Simple = LL or worse at decode sizes, measured).
- **bottleneck**: all-reduce, 23% of wall (decode 6.2 ms/step, prefill 65 ms/step).
- **upside**: decode: 135 x (57-30) us = -3.6 ms/step (-10% decode) ; prefill steps: -30..-40 ms of 195 ms. Total
  **~+10-15%** if the kernel reaches ~20 GB/s effective; medium confidence on reaching it, the biggest single lever in
  the profile. Risk: flag polling over PCIe adds latency, and the kernel occupies SMs (decode-step kernels are serial
  anyway, so that's fine).
- **novelty**: untried. Different bottleneck from idea 1 (comm vs drafter head), so the gains add.
- **cheapest test**: standalone 2-process microbench (no vLLM) of the kernel vs NCCL at T=16/48/400/2048 x 5120 bf16
  inside a CUDA graph; go/no-go if T=48 < 40 us. ~3 GPU-min. Only then integrate.

## Idea 3: Greedy target sampling without the logits all-gather
- **direction**: When every request in the batch is greedy (temperature 0) and asks for no logprobs / penalties /
  logit bias / structured output, compute the target's argmax vocab-parallel (local max+index per rank, all-gather
  2 values per token, lowest-index tie-break) instead of all-gathering 48x248320 logits; fall back to the full path
  otherwise. Generic (keyed to request params, not to the benchmark). Exact: same argmax by construction.
- **why**: measured: the target's logits AllGather clusters at ~950 us per decode step (~280 launches ~875-950 us in
  the trace, ~1 per step), plus elementwise work on full logits. Same mechanism already exists for the drafter
  (`get_top_tokens`), so the building block is in-tree.
- **bottleneck**: ~1 ms of 34.8 ms decode step (~2.7%).
- **upside**: **+2-3%**, high confidence on the mechanism; moderate integration work in the V2 sampler + rejection
  sampler (greedy path compares target argmax to draft ids).
- **novelty**: untried; adds to ideas 1 and 2 (different kernels). Natural to bundle with idea 1 in one team.
- **cheapest test**: code it, verify bitwise-equal sampled ids vs the full-gather path on recorded hidden states,
  then one A/B bench.

## Idea 4: Marlin decode efficiency (weight GEMMs at 64-84% of DRAM peak)
- **direction**: Marlin is 50% of all kernel time and ~17.5 ms of a decode step. At M=48 it runs 128 blocks (1/SM,
  8% warp occupancy) and is limited by `math_pipe_throttle` (FP8->bf16 dequant + MMA on few warps), not DRAM. Try
  Marlin's other tile configs / 2 blocks per SM for M<=64, and for the small GEMMs (16 MB out_proj at 630 GB/s) split-K
  or fusing GEMMs that share an input (e.g. GDN in_proj_qkvz with in_proj_ba). Same math (weight-only FP8, bf16
  compute, fp32 accumulate).
- **why**: measured ncu: 804/630/795/770 GB/s for the 4 GEMMs of a layer; 4090 peak ~1008 GB/s. Per-layer GDN weights
  192 MB in 255 us = 753 GB/s average.
- **bottleneck**: Marlin, ~50% of decode-step time.
- **upside**: reaching ~880 GB/s average = -2.5 ms/step = **~+5% decode, ~+3-4% total**; low-medium confidence
  (Marlin is a mature kernel; tile heuristics may already be near-best on this shape).
- **novelty**: untried. Touches a different bottleneck from 1-3, gains add.
- **cheapest test**: microbench marlin_gemm with the 4 real shapes at M=16/48 over the available configs (no server):
  ~2 GPU-min; if the best config beats the default by >10% it is worth a heuristic change.

---

## Bets
1. **Idea 1 (drafter head diet + depth)**: cheapest (flags only), clean evidence (10% of decode in the drafter head),
   allowed explicitly; a team can also bundle idea 3 into it. Most likely to produce an official win this round.
2. **Idea 2 (host-staged all-reduce)**: largest measured gap (AR 23% of wall, ~2-2.5x off the link's speed), but a real
   kernel project; worth one team for the round, with a go/no-go microbench in the first 30 minutes.
Combinations: 1+3 share the lm_head/logits path (additive, different kernels); 2 is orthogonal to both; 4 is
orthogonal but lower confidence, a backup direction.
