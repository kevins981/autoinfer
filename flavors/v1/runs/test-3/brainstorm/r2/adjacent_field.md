# Round 2 ideas: bs-r2-adjacent_field (databases / HPC view)

Scripts: `agents/bs-r2-adjacent_field/a2.py … a8.py` (stdlib sqlite queries). Unless noted, every number
comes from the **champion T1-005 nsys** (`workspaces/leader-1.hostar2/out/nsysB/profile.sqlite`, GPU 0,
64 prompts at c=16, steps split at `_prepare_pos_seq_lens_kernel`).

## What I measured first
**D1. Where the champion's time goes** (25.4 s window, 549 steps)
- **Decode-only steps:** 497 steps, 16.8 s (66%). With 16 sequences a step takes 38.5 ms (step 300).
- **Prefill-containing steps:** 51 steps, 8.65 s (34%).

Budget of one decode step with 16 sequences (38.5 ms, step 300), measured:
| part | time | share |
|---|---|---|
| target layers (64 layers, 1.69 → 30.9 ms) | 29.2 ms | |
| Marlin, inside those layers | ~18.3 ms | 47% |
| host AR, 128 × ~44 µs | 5.6 ms | 15% |
| GDN decode kernel, 48 × ~77 µs | 3.7 ms | 10% |
| **lm_heads + logits all-gathers** | **6.0 ms** | **16%** |
| gap at step start (see D4) | 1.7 ms | |

The lm_heads + logits all-gathers break down as:
- target lm_head (`ampere_bf16_s1688gemm 128x64`): 1.59 ms;
- target logits all-gather: **0.96 ms**;
- 2 draft lm_heads (`Kernel2`): 1.37 ms each;
- 2 draft logits all-gathers: 0.36 ms each.

**D2. GDN spec-decode state traffic is about a third the size of the weight traffic.**
- Code: `csrc/gdn/fused_gdn_decode_kernel.cu:199-203,321-335`. The kernel reads **one** fp32 state per
  (request, value head) and writes **one full state per token**, for all 3 verified tokens (`destination_slot > 0`).
- Per-rank state size (config: 48 value heads, so 24 per rank; 128×128; `mamba_ssm_dtype=float32`):
  1.57 MB per sequence per layer, i.e. 25 MB per layer at 16 sequences.
- So each GDN layer moves **25 MB of reads + 75 MB of writes = 100 MB**, against 196 MB of Marlin weights.
  Over 48 GDN layers that is **4.8 GB/step of state traffic** next to ~13 GB of weights.
- The kernel takes 77 µs. Moving 100 MB in 77 µs would need 1.3 TB/s, which is more than the 4090's DRAM
  can do. So the writes sit dirty in L2 and are written back during the **next kernels**. That spill is
  measured, comparing identical-shape GEMMs after a GDN layer and after an attention layer, both in step 300:
  - gate_up (`Marlin<256,3,16,4>`, 91 MB): **141 µs after GDN vs 122 µs after attention** (+19 µs);
  - out_proj vs attention o_proj (both 3072×5120, 15.7 MB): 32–33 µs vs 25.6–27 µs (+6 µs).

  The GDN state traffic costs **~104 µs per GDN layer, i.e. ~5 ms per 16-sequence decode step (13%)**.
- Of the 3 written states, only one is ever read again (`source_slot = state_indices[accepted-1]`). The
  other two (≈50 MB/layer) are write-only waste.

**D3. The "unexplained" prefill-step AR skew is a hardware asymmetry: GPU 0 is ~15% slower on compute-bound kernels.**
- Kernel time inside prefill steps, GPU 0 vs GPU 1, same kernels and same shapes (a6.py):
  | profile | Marlin GPU 0 | Marlin GPU 1 |
  |---|---|---|
  | BASE `task/baseline/nsys_c16` | 4595 ms | 3973 ms |
  | T1-002 | 4611 ms | 3988 ms |
  | T1-005 | 4647 ms | 4001 ms |

  That is **+15.6% on GPU 0, in all three profiles, whatever the comm backend**. Nearly every other prefill
  kernel is also 5–18% slower on GPU 0.
- In prefill steps, **rank 0 arrives last at 96% of ARs** (mean start skew 186 µs). In decode, rank 0 is
  only ~1.7% slower on the memory-bound gate_up (141.0 vs 138.6 µs median), and skew is ~4 µs.
- `nvidia-smi` (idle): same VBIOS 95.02.3C.C0.F4 and the same 450 W limit on both GPUs. Both GPUs log
  "SW Power Capping" time.
- Interpretation: compute-bound prefill runs into the power cap, and GPU 0's silicon clocks lower at 450 W.
  This is the source of KB3/KB4's skew. Software can't remove it: the fix would be power limits or clocks,
  which need root. Not tested: clocks logged during a GEMM loop on each GPU (≈1 GPU-min would confirm).
- **Implication: drop "find the skew source" as a direction.** Uneven TP shards (give GPU 1 more) don't pay
  either. Napkin: balancing prefill (f = 0.464) costs decode about 2.3% of GEMM time, because decode is
  memory-bound and nearly symmetric. Weighted 66/34, the net is ≈ 0.

**D4. The 1.25 ms gap at the start of each decode step is CPU time spent in `cudaGraphLaunch`** (1.38 ms per
step, from the RUNTIME table). The profile was taken with `--cuda-graph-trace=node`, which instruments every
node of the ~900-node graph. **Most likely an nsys artifact; nobody should chase it** without first timing it
outside nsys.

**D5. Prefill Marlin is not that inefficient.**
- About 22K tokens went through the prefill-step GEMMs (19,755 input tokens from bench.log + ~2.3K decode
  tokens). At ~24.5 GFLOP/token/rank that is 539 TFLOP in 4.39 s, i.e. **~123 TFLOPS against the 165 TFLOPS
  bf16/fp32-accumulate peak (75%)**.
- So the "dequantize, then cuBLAS for large M" idea would buy ≤ 2% overall. Not proposed.

---

## Idea 1 (my top bet): checkpoint + log replay for the GDN spec-decode state (database WAL, applied to the recurrent state)
- **direction:** Today, every verified token writes a full 64 KB-per-head fp32 state "page image", and 2 of
  the 3 are thrown away. Instead, do what a database does with uncommitted transactions:
  - Per step, persist **one checkpoint**: the state after token 0, which is always accepted.
  - For tokens 1..n-1, write a tiny **log record** per head: `k_t` (fp32, 128), `delta_t = (v_t − h·k_t)·β_t`
    (fp32, 128) and `decay_t`. That is about 1 KB instead of 64 KB. It can live in the first KB of the
    slots those tokens already own, so there is no new allocation.
  - The next step's kernel already reads `num_accepted_tokens`. It loads the checkpoint and **replays the
    accepted log records** with the kernel's own update code (`h *= decay; h += k*delta`, same order). It
    then continues with the new tokens and writes the next checkpoint in place.
  - Any other consumer of a slot must call a small "materialize" kernel first: prefix-cache align copies
    (`postprocess_mamba_*`, block size 1600 tokens, so rare), preemption, and a prefill continuation.
- **exactness:** the replay uses the same fp32 values in the same operation order, so the states are
  **bitwise identical**. Outputs `o_t` are still computed in registers, as today. A greedy-output equality
  test on N prompts checks it.
- **why:** D2. The traffic per GDN layer goes from 100 MB to ≈50 MB (25 read + 25 write + ~0.1 MB of log).
- **bottleneck:** GDN state traffic, ≈104 µs per GDN layer (77 µs in the kernel + ~25 µs spilled into the next
  GEMMs). That is **~5 ms of the 38.5 ms decode step (13%)**, measured.
- **upside (napkin):** 50 MB at ~0.8 TB/s ≈ 62 µs per layer, against 104 µs now. That saves ~42 µs × 48 ≈
  **2.0 ms/step (≈5% of a decode step)**, plus the decode tokens inside prefill steps. That is **+3.5 to +5%
  tok/s**. Confidence: medium. The traffic and the spill are measured; the post-change kernel speed is assumed.
- **bonus:** the cost of each extra speculative token in GDN drops from a 25 MB write per layer (1.2 GB/step
  ≈ 1.4 ms) to near zero. That re-opens **num_spec_tokens=3**, which round 1 ruled out at ~+4% in noise.
- **novelty:** this is R1-K (SE6, "write one state per step, replay accepted tokens"). It was scored U2
  and never tried. New here: the spill measurement (gate_up 141 vs 122 µs) and the 4.8 GB/step traffic
  figure put the upside at 2× R1-K's 9%-of-step estimate. The log-in-the-existing-slot layout also
  avoids new allocations.
- **effort / risk:** medium-high. About 60 lines of kernel code and an audit of the slot consumers
  (mamba align and prefix caching, preemption, chunked prefill). The scheduler is untouched: slot ids and
  `num_accepted_tokens` already exist.
- **cheapest test (no engine change, ≈1 GPU-min, 1 GPU):** a microbenchmark that calls
  `torch.ops._C.fused_gdn_decode_post_conv_mtp` at the real shapes (16 requests, 3 tokens, 24 heads, fp32
  state), followed by a 91 MB-read kernel (or the Marlin gate_up). Run it two ways:
  - (a) `state_indices` with 3 slots per request;
  - (b) the same with columns 1 and 2 set to 0, so the kernel skips those writes (line 325).

  The time saved by (b) is the ceiling of this idea. Go if (a−b) ≥ 30 µs per layer.

## Idea 2: the "logits diet" bundle (predicate pushdown: ship the argmax, not the table)
- **direction:** One round for the parts of the 6.0 ms/step lm_head + logits block (16% of decode) that
  never change the output:
  1. `use_local_argmax_reduction: true` in the speculative config. It exists
     (`vllm/config/speculative.py:540`) and is wired into MRV2 (`spec_decode/speculator.py:181`). It removes
     the 2 × 0.36 ms draft logits all-gathers.
  2. A **greedy target-verification fast path**: when every request in the batch is greedy with no
     logprobs, penalties or logit processors, each rank sends (max, argmax) per row instead of 48×124K
     bf16 logits. That saves ~0.9 ms. Exactness: on ties, the lowest global id wins, which is what
     `torch.argmax` does on the full row.
  3. A **cheaper draft head**. `draft_lm_head_quantization: "fp8"` is supported for MTP
     (`mtp/config.py:21`) and allowed by ruling 1. It halves 2 × 1.37 ms. Optionally add the reduced-vocab
     prefix (ruling 2; frequencies from non-benchmark text) on top, for ≈ 8× fewer bytes.
- **why:** D1. All four numbers are measured in step 300.
- **upside (napkin):** (1) 0.7 ms + (2) 0.9 ms + (3) ~1.4 ms (fp8) ≈ **3 ms/step ≈ 8% of a decode step,
  ≈ +5% tok/s**. Subtract the acceptance loss of (3): unknown, probably 1–3%. Confidence: high for (1) and
  (2), medium for (3).
- **novelty:** low. These are R1-F/G/I/J, all untried. Bundled and with rulings now in hand, they fit one
  team-round, and the risk is low.
- **cheapest test:** (1) and (3) are config flags: one same-job A/B bench against T1-005 (≈ 8 GPU-min).
  Also check the acceptance length in the bench log. (2) needs a greedy-equality check on 50 prompts.

## Idea 3 (secondary): copy-engine, full-duplex AR for large M (prefill steps)
- **direction:** For M above ~256 rows, replace the SM-driven host-staged AR with a chunked copy-engine
  pipeline:
  - per ~1 MB chunk: `cudaMemcpyAsync` D2H of the own partial into the shm slot, then a
    `cuStreamWriteValue32` flag;
  - the peer `cuStreamWaitValue32`s on that flag and copies the chunk H2D;
  - a small add kernel finishes each chunk.

  Chunk i+1 goes up while chunk i comes down, so both PCIe directions stay busy. It is graph-capturable
  (memcpy nodes and stream memory ops).
- **why:**
  - Measured in round 1 (brainstorm/r1/adjacent_field.md M2): copy engines run **20.1 GB/s in each direction
    at the same time**.
  - KB5: SM stores and loads over PCIe only partly overlap. The large-M AR reaches ~12–14 GB/s.
  - On GPU 0, the late rank, so its AR time is mostly transport: prefill-step hostar is 2.42 s of 25.4 s
    (9.5%). That is ~367 µs per AR at a mean M of ~430 rows (4.4 MB). The floor is ≈ 225 µs.
- **upside:** ≈ 100 µs × ~129 ARs × 51 steps ≈ 0.65 s, i.e. **+2 to +3%**, plus better TTFT and ITL p99.
  Confidence: medium-low (chunk latency overheads).
- **novelty:** this is leader-1's own open lead. It is listed so the manager can pair it with Idea 1 or 2
  if a team has spare capacity.
- **cheapest test:** extend `brainstorm/r1/adjacent_field_mb/ar_bench.py` with the CE pipeline at M=430 and
  M=2048 (≈2 GPU-min). Go if ≤ 280 µs at M=430.

## Not proposed (with numbers), so nobody spends a round on them
- **Chasing the prefill skew, or uneven TP:** D3. It is GPU 0 hardware (+15.6% compute time in all profiles), and the net of rebalancing is ≈ 0.
- **The decode step-start gap (1.25 ms):** D4. It is `cudaGraphLaunch` under node tracing; time it outside nsys first.
- **Dequantize + cuBLAS for prefill GEMMs:** D5. Marlin is already at ~75% of peak there, so ≤ 2%.

## My bets
1. **Idea 1 (GDN checkpoint + log replay).** It is the largest new, measured waste: 50 MB of write-only
   traffic per layer, ~2 ms/step. It is exact (bitwise), it has a zero-code 1-GPU-min ceiling test, and it
   makes a 3rd draft token cheaper.
2. **Idea 2 (logits diet bundle).** Lowest risk and mostly flags. ~+5% is plausible and it stacks with
   Idea 1, since they touch disjoint time. A good second team.
3. Idea 3 only if a third slot exists.

Note for everyone: D3 (rank-0 slowness) means the remaining prefill AR "skew" is a hardware floor; KB3/KB4 can be updated.
