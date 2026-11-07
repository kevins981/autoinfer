# Round 2 ideas: systematic engineer (bs-r2-systematic_engineer)

Every "measured" number comes from my queries of the champion T1-005 nsys profile
(`workspaces/leader-1.hostar2/out/nsysB/profile.sqlite`, 64 prompts at c=16) or from my own GPU job.
Scripts are in `brainstorm/r2/systematic_engineer_queries/` (run them from the run folder):
- s1: per-step breakdown;
- s2: the kernels of a full 16-request decode step;
- s3: rank skew at each AR and what causes it;
- s4: the same kernel's duration on GPU 0 vs GPU 1 (s4_base / s4_t2: the same query on BASE and T1-002);
- s5: concurrency check.

My clock job is in `agents/bs-r2-systematic_engineer/clk/` (clk.out, smi.csv).

## Where the time goes in the champion (measured, s1/s2, GPU 0)
Profile wall time is 25.4 s.

**Decode-only steps: 66% of wall time.** A full 16-request step (48 tokens) has a median of
38.5 ms, with 36.6 ms of GPU busy time:

| part | per full step | notes |
|---|---|---|
| Marlin | 18.7 ms | gate_up: 65 x 136 us = 655 GB/s for 89 MB. The other shapes: 195 x 49 us, ~670 GB/s. |
| host AR | 5.9 ms | 137 x 43 us, grid = **8 blocks** |
| logits all-gathers | 1.67 ms | target 967 us + 2 drafts x 354 us |
| lm_head | 4.33 ms | target at M=48: 1593 us (798 GB/s). 2 draft heads at M=16: 1368 us each (929 GB/s, at the floor). Each reads this rank's 1.27 GB bf16 shard (124160 x 5120). |
| GDN MTP decode | 3.65 ms | 48 x 76 us |
| other | ~1.5 ms | |
| idle | 1.9 ms | |

- Small batches: 1-3 requests take 25.6 ms per step. Marlin is still 15.3 ms there (~800 GB/s)
  and lm_head 4.8 ms.
- Over all decode steps (16.8 s): Marlin 8.8 s (**35% of total wall**), AR 2.66 s, lm_head
  2.5 s, GDN 1.2 s.

**Steps containing a prefill: 34% of wall time.**
- Step wall time fits 59.6 ms + 0.284 ms per prefill token.
- Totals over these steps (8.66 s): Marlin 4.65 s (18% of total wall), AR 2.42 s (9.5% of
  total wall), GDN 0.42 s.
- Large-M Marlin runs at ~120 TFLOPS on GPU 0 (M ~ 950: 188.6 ms per step). That is close to
  cuBLAS's 143.6 TFLOPS on that GPU (measured, clk.out), so prefill GEMMs are near their roofline.

**New finding: GPU 0 is clock-locked, and that is where the "unexplained" prefill rank skew
(KB3/KB4) comes from.** Measured:
- clk job, smi.csv: GPU 0 stays at **2235 MHz SM even idle and under load** (P0/P2, 370 W, no
  throttle reason). GPU 1 idles at 210 MHz and boosts to 2370-2415 MHz (450 W, power-capped).
  Something outside our control has locked GPU 0's clocks; resetting them needs root.
- bf16 8192³ GEMM: GPU 0 143.6 TFLOPS, GPU 1 153-157 TFLOPS. Copy bandwidth is equal
  (914 / 920 GB/s).
- In the server (s4), the same kernels take longer on GPU 0. Ratio of time on GPU 0 to GPU 1:

  | kernel | GPU 0 / GPU 1 |
  |---|---|
  | large-M Marlin | 1.17 |
  | flashinfer prefill attention | 1.15 |
  | fused_add_rmsnorm | 1.15 |
  | act_and_mul | 1.13 |
  | decode Marlin (memory-bound) | 1.02-1.04 |

  Same ratios in BASE (s4_base: 1.169) and T1-002 (s4_t2: 1.168). It is not a one-off.
- s3: at the 4772 large prefill ARs, GPU 0 arrives last 92% of the time. Mean |skew| is 268 us,
  and the per-segment busy-time difference (-132 us) is mostly Marlin. Nothing runs
  concurrently on GPU 0 (s5).
- Interpretation: compute- and latency-bound work is gated by the slower GPU 0. The skew costs
  4772 x 268 us ≈ 1.28 s in prefill ARs, plus 72K x 6.8 us ≈ 0.49 s in decode ARs. That is **~7%
  of wall time spent waiting for GPU 0**. The leaders and the human should know: no transport
  fix can remove it. Unlocking GPU 0's clocks would be a hardware change: not ours to make, and
  BASE was measured with the lock too. I flag it for the manager/human; it is not a direction.
  (The obvious software answer, an uneven TP split, nets only ~+1.5% by my napkin math, because
  decode is memory-bound and equally fast on both GPUs. Not proposed.)

---

## Idea 1: Logits diet: draft heads in FP8 + local-argmax reductions for drafts and greedy verification (+ later a reduced-vocab draft head)
- **Direction:** cut the per-step lm_head and logits traffic, which never got a round.
  - Step 1 is two existing flags in the speculative config (`vllm/config/speculative.py:534,540`;
    MTP implements `get_top_tokens`, `qwen3_5_mtp.py:310`; V2 model runner, which the server
    uses):
    `"draft_lm_head_quantization":"fp8"` (human ruling 1: allowed) and
    `"use_local_argmax_reduction":true`.
  - Step 2 is code: a greedy fast path for the target's verification. When every request in the
    batch is greedy, has no logprobs and no logit processors, use
    `LogitsProcessor.get_top_tokens` (`logits_processor.py:~225`: local max, then an all-gather of
    (value, index) pairs) instead of all-gathering the full 48 x 248K logits. Fall back to the
    full path otherwise.
  - Step 3, optional and acceptance-dependent: a reduced-vocab FP8 draft head (ruling 2).
- **Why (measured, s2, per full decode step):**
  - the draft heads read 2 x 1.27 GB (2 x 1368 us);
  - the draft logits all-gathers cost 2 x 354 us;
  - the target logits all-gather costs 967 us, though greedy verification only needs the argmax.
- **Exactness:**
  - The verification path is the same values. The cross-rank max with first-index tie-break
    equals torch's argmax over the concatenated logits, because rank 0 holds the lower ids. The
    float32 index is exact below 2^24.
  - The FP8 draft head changes only draft proposals; verification stays the target's bf16 head.
- **Bottleneck:** ~3.0 ms of a 38.5 ms step (fp8 heads save ~1.37 ms, draft local argmax ~0.7 ms,
  target local argmax ~0.95 ms). Mixed steps carry the same costs (lm_head 5 ms per prefill step,
  measured).
- **Upside:** ~7.8% of decode step time; decode is 66% of wall, plus the mixed-step share.
  Estimate **+5-6% throughput** (napkin). Confidence medium-high: the bytes and the all-gathers
  are measured, and the flags exist. Risks:
  - the FP8 head may cost a little acceptance (now 2.34-2.40 mean; per-position 0.80/0.605,
    server.log);
  - its extra 0.64 GB per rank must fit (memory change → the 128K test).
  - Step 3 adds ~+1 ms/step (~2%) if acceptance holds.
- **Add-on, once it works:** re-check 3 draft tokens. Position 2's conditional acceptance is
  0.757 (measured). A third position at ~0.72 (assumed) gives length 2.40 → ~2.84 (+18%) for a
  step that is ~13% longer (napkin: M 48→64, one more draft + GDN state). That is ~+4% net, and
  better once drafts are cheap. Also a memory change (128K test).
- **Novelty:** proposed in r1 (R1-F/G/I/J), never tried. The rulings now allow it.
- **Cheapest test:** one GPU job, same-job A/B: champion vs champion + the two flags, 400-prompt
  bench, and read the acceptance from server.log. Then the target fast path (code) plus a GSM8K
  check (it should be bitwise: compare outputs on 50 prompts).

## Idea 2: Fill the AR windows: prefetch the next GEMM's weights into L2 while the host-staged AR runs
- **Direction:** the host AR kernel occupies **8 of 128 SMs** (grid=8, measured) for 43 us,
  137 times per decode step. DRAM and 120 SMs sit idle meanwhile.
  - On a side stream forked and joined inside the CUDA graph, launch a read-only kernel that
    touches the first N MB of the next Marlin weight, so the GEMM then hits L2 (72 MB):
    - after the attention/GDN AR → gate_up (89 MB): its first ~35-40 MB;
    - after the MLP AR → the next layer's in_proj (42 MB) / qkv (37 MB).
  - Math is unchanged and there is no data hazard (read-only weights).
- **Why:** 5.9 ms/step of AR windows (measured), with ~43 us x ~0.9 TB/s ≈ 39 MB fetchable per
  window (napkin). Decode Marlin is 35% of total wall and runs at 655-670 GB/s (measured).
- **Bottleneck:** decode Marlin DRAM time, using the AR's idle time.
- **Upside:** if the GEMM saves 15-30 us per window (assumed: L2 hits, minus Marlin's own
  compute limit at M=48), that is 2-4 ms/step = 5-10% of the decode step → **+3-7% throughput**.
  Confidence low-medium. Unknowns:
  - Ada L2 retention for ~40 MB;
  - whether Marlin's cp.async hits L2 as hoped (its evict hints);
  - whether Marlin at M=48 is partly issue-bound (ncu: tensor pipe 26%, stalls
    math_pipe_throttle), which would cap the gain.
- **Novelty:** R1-D, scored N5 but never tried. The host AR (8 SMs, spinning on PCIe) makes the
  window cleaner than NCCL's was.
- **Cheapest test (kill switch, ~1 GPU-min):** microbench Marlin gate_up at M=48 with real
  shapes, cold L2 vs after prefetching 20/40 MB. Also check that the 491 KB host AR is not slowed
  by a concurrent prefetch kernel. Drop the idea if the GEMM gains <15 us.

## Idea 3: Marlin at decode M=48: close the gap between 670 and ~800-900 GB/s
- **Direction:** the same Marlin streams the same 12.2 GB at ~800 GB/s with 1-3 requests
  (15.3 ms) but at ~655-670 GB/s at M=48 (18.7 ms) (measured, s1/s2). The extra ~3.4 ms per step
  is in-kernel compute/issue overhead, not DRAM.
  - Sweep Marlin's per-shape config (threads, thread_n/k, stages, fp32-reduce, more blocks per
    SM through fewer stages) on the 5 real shapes at M=16/32/48/64.
  - Hardcode the sm_89 winners.
  - If configs don't do it: a split-K or 2-blocks-per-SM variant. Today it runs 1 block per SM
    with 4-8 warps; occupancy is limited by ~101 KB of smem (ncu key csv).
  - Same math up to accumulation order.
- **Why:** decode Marlin is 35% of total wall (measured). M=3 sets a measured in-kernel bar of
  ~800 GB/s, and 914 GB/s copy bandwidth is the ceiling (measured).
- **Upside:** M=48 at 800 GB/s saves 3.4 ms/step (8.8% of decode) → **~+5%**; at 900 GB/s ~+8%.
  Confidence: medium for the config part, low for a new kernel variant in one round.
- **Combination:** overlaps with idea 2 (both target decode Marlin): if Marlin is issue-bound,
  idea 3 is the precondition for idea 2's full gain. Both add to idea 1, which touches lm_head
  and comms.
- **Novelty:** R1-E, untried. The new data point is the M=3 vs M=48 contrast at identical
  bytes, which locates the loss inside the kernel.
- **Cheapest test:** a GPU-only microbench of the shapes and configs, ~5 min, no server.

## Idea 4 (lower): GDN MTP decode writes 3 recurrent states per step
- R1-K unchanged: 3.65 ms/step (9.5% of decode, measured), est. ~1.5 ms saved → ~+2.5%. High
  effort. Do it after ideas 1-3.

## Ruled down with numbers
- **Copy-engine AR for large M:** the prefill AR is 2.42 s, but ~1.28 s of it is GPU 0 skew
  (above), which transport can't fix. Going from 13 to 20 GB/s saves ≲0.4 s (~1.5%).
- **Prefill AR/compute overlap (R1-L):** hides at most the ~1.1 s of transport (≤4%), with heavy
  engineering. Skew remains.
- **Prefill GEMM kernel swap:** Marlin is already at ~85% of cuBLAS at large M on GPU 0
  (measured). Small-M mixed steps (M ~100: 35 ms Marlin vs 18.7 ms at M=48) are a real but
  smaller pocket, ~2%.

## My bets
1. **Idea 1.** The biggest certain, measured win per GPU-minute. Two flags plus one well-scoped
   code path, exact, ~+5-6%, testable in one job. Then the 3-draft re-check.
2. **Idea 2 with idea 3 as its partner** (same bottleneck: decode Marlin, 35% of wall). Start
   with the 1-GPU-min L2 kill test and the Marlin config sweep. Together they have the largest
   remaining upside (+5-10%), but less certainty.
3. Tell the human about GPU 0's clock lock: ~7% of wall is spent waiting for it, and no
   software direction recovers that cheaply.
