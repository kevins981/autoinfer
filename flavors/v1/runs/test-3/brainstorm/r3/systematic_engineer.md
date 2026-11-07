# Round 3 ideas: systematic engineer (bs-r3-systematic_engineer)

New measurements this round (all reproducible):
- **M1, Marlin FP8 microbenchmark**: the real per-rank decode shapes, group 128, cold weights
  (rotating copies, 200 MB per shape), CUDA graph of 40 calls, M = 1..96, both GPUs (one job,
  ~3 GPU-min). Files: `agents/bs-r3-systematic_engineer/mb/` (mb.py, gpu0.out, gpu1.out).
- **M2, in-server decode Marlin per call**, grouped by shape and by the kernel before it. Source:
  the T1-005 nsys profile, 192 full 16-request decode steps, GPU 0. Files:
  `brainstorm/r3/systematic_engineer_queries/q1_marlin_inserver.py`, q1.out.
- **M3, k=3 diagnosis from the existing logs**: the scheduler stats lines in
  `workspaces/leader-1.logitsflags/out/{C2b,C3}/server.log` (T1-008 vs its same-job T1-007 rerun).
- **M4, code reading**: the GDN spec-decode kernel and vLLM's Mamba "align" block accounting.

## Headline findings

### F1. The k=3 TTFT blow-up (T1-008, KB11) is a capacity limit, not compute (measured, M3)
- Each request holds a fixed number of GDN-state pages, whatever its length. At k=2 the pool
  holds 16 requests; at k=3 it holds only 14, so 2 requests always wait.
- Scheduler lines during the steady part of the benchmark:
  - k=2 (C2b): `Running: 16 reqs, Waiting: 0, KV usage 83.8%` (5.2-5.3% per request).
  - k=3 (C3): `Running: 14 reqs, Waiting: 2 reqs, KV usage 95.3%` in every line (6.8% per
    request).
  - The champion's official server shows the same 83.8% at 16 requests.
- Why (code, M4):
  - Prefix caching is on, so the Mamba cache mode is "align" (server.log: "Mamba cache mode is
    set to 'align'").
  - A GDN request can then hold up to `2 + num_speculative_blocks + 1 checkpoint` state pages
    (`kv_cache_interface.py:1140`; `qwen_gdn_linear_attn.py:904`).
  - One page set is the fp32 state of 48 GDN layers: 24 heads x 128 x 128 x 4 B = 1.57 MB per
    layer per rank, so ~75 MB per state.
  - At k=2 that is ~281 MB per request (5.2% of 5 GiB). At k=3 it is ~365 MB (6.8%), and
    16 x 6.8% = 109% of the 5 GiB budget.
  - The attention block size is forced to 1600 tokens to match the page size. ShareGPT requests
    need just one attention page each, so the request count, not the token count, fills the pool.
- Interpretation: T1-008 ran with 14 slots for 16 clients. The 2 waiting requests explain
  TTFT p50 210 → 792 ms and the lower throughput. Its decode TPOT still fell 6%.
- **Not a step-cost problem.** "3 draft tokens hurt TTFT" in KB11 should be re-read as "3 draft
  tokens don't fit in a 5 GiB KV budget".

### F2. In the server, decode Marlin runs ~2.5 ms/step slower than the same shapes in isolation, and most of the excess follows the GDN layers (measured, M1 vs M2)

| shape (per rank) | isolated, M=48, GPU 0 (M1) | in server, M=48 (M2) | excess per step |
|---|---|---|---|
| gate_up 5120x17408 | 113.9 us | 134.4 avg (median 141.1; ~141 after GDN layers, ~122 after attention layers, KB9) | **+1.33 ms** |
| GDN in_proj 5120x8192 | 50.6 | 59.1 | +0.41 ms |
| GDN out_proj 3072x5120 (right after the GDN kernel) | 24.5 | 32.5 | +0.38 ms |
| down 8704x5120 | 56.7 | 60.1 | +0.22 ms |
| attn qkv / o | 46.9 / 24.4 | 52.6 / 26.2 | +0.13 ms |
| **sum** | 15.66 ms/step (M1 total, 64 layers) | ~18.2 ms | **~2.5 ms** |

- Interpretation:
  - The GDN spec-decode kernel writes 3 fp32 states per layer per step (25 MB read + 75 MB
    written per layer per rank).
  - The kernel takes 76 us (KB9), which is faster than 100 MB can move at ~0.9 TB/s (111 us).
    So the writes land in L2 as dirty lines and drain during the next GEMMs.
  - The excess on the GEMMs right after GDN (~+8 us on out_proj, ~+27 us on the following
    gate_up) closes that gap: 76 + ~35 ≈ 111 us. That matches the state traffic at full DRAM
    bandwidth.
  - So GDN decode really costs ~5.3 ms/step (3.65 ms in-kernel + ~1.7 ms spilled into Marlin).
    It is purely state-bandwidth-bound.

### F3. Marlin's in-kernel M=48 penalty is small (measured, M1). This downgrades R2-D
- Per step (64 layers, no lm_head), on GPU 0:

  | M | Marlin time per step |
  |---|---|
  | 1 | 14.00 ms |
  | 3 | 14.03 ms |
  | 16 | 14.38 ms |
  | 32 | 14.81 ms |
  | **48** | **15.66 ms** |
  | **64** | **17.22 ms** |
  | 80 | 24.67 ms |
  | 96 | 29.42 ms |

- GPU 1 is ~3% faster: 15.21 ms at M=48, 16.34 ms at M=64.
- fp32-reduce off: 15.45 ms at M=48.
- What this means for R2-D:
  - R2-D's premise ("670 vs 800 GB/s, the loss is inside the kernel") is only half right. The
    in-kernel cost of M=48 over M=3 is **1.6 ms**. The other ~2.5 ms is the interference in F2.
  - A config retune can recover at most ~1.6 ms, realistically ~0.5-0.8 ms (≈1.5-2%). That is
    not worth a round on its own.
- Also measured: **M=64 (k=3 at 16 requests) costs only +1.56 ms over M=48. M=80 (k=4) hits a
  cliff (+9 ms; gate_up 466 GB/s).** k=4 is ruled out; k=3 is the ceiling.

---

## Idea 1 (top bet): GDN spec-decode "checkpoint + replay": one state read and one state write per layer per step, instead of one read and k+1 writes
- **Direction:** change the GDN spec-decode path (`third_party/flash_linear_attention/ops/
  fused_recurrent.py`, `IS_SPEC_DECODING`; stores at lines ~153-166).
  - Today the kernel stores the full fp32 state after every one of the k+1 tokens, so that next
    step it can load the one that was accepted (`num_accepted_tokens - 1`).
  - Instead, keep one committed state per request plus a small per-request ring of the last
    step's per-token GDN inputs (q, k, v, g, beta after the conv: ~10-15 KB per token per layer).
  - Each step the kernel:
    1. loads the committed state;
    2. replays last step's accepted tokens (≤ k+1 recurrence steps, in registers);
    3. stores that as the new committed state (the only state write);
    4. runs the k+1 new tokens to produce their outputs, without storing their states.
  - The conv state needs the same treatment (it is small).
  - First version: keep the block accounting unchanged and only drop the stores (a pure speed
    change). Second version: shrink `num_speculative_blocks` for GDN. That frees ~150 MB per
    request (2 pages) and removes F1's capacity limit.
- **Exactness:** bitwise. The replay runs the same fp32 recurrence, in the same order, on the
  same inputs as the step that originally produced those states. Verification outputs are
  unchanged.
  - Check: state tensors bitwise-equal to today's kernel after random accept patterns, in a unit
    test, plus GSM8K and acceptance.
  - The prefix-cache "align" checkpoints at 1600-token boundaries must still be materialized:
    replay writes the committed state, so the align logic must take its checkpoint from that
    state. This is the main integration risk.
- **Why (measured):** GDN decode costs ~5.3 ms of a full decode step (F2: 3.65 ms in-kernel per
  KB9 + ~1.7 ms spill measured in M2). Its traffic is 100 MB per layer at k=2 and would be
  50 MB per layer after the change (state read + one write; the ring is negligible).
  - The replay compute is assumed negligible: ~0.2 GFLOP per layer per step at k=2, a few us of
    fp32.
- **Bottleneck:** GDN state bandwidth plus its L2 spill into the next GEMMs. That is ~14% of a
  T1-009 decode step (assumed ~35.5 ms: the measured 38.5 ms of T1-005 minus the ~3 ms logits
  diet).
- **Upside:**
  - Halving the traffic saves ~55 us x 48 layers ≈ **2.6 ms per decode step** (7% of the step).
  - Mixed steps also run the decode GDN path for their decode rows.
  - Estimate: **+4.5-6% throughput at k=2.** At k=3 the state traffic would be 125 MB per layer
    today, so the saving grows to ~3.9 ms per step.
  - Confidence: medium. The bandwidth accounting is consistent three ways (F2), but the replay
    kernel must stay bandwidth-bound and the align-mode integration is real work.
- **Novelty:** R1-K/R2-B, never selected. What is new:
  - F2: the cost is ~45% larger than the kernel time alone, because of the measured spill.
  - F1: the same spec pages are what cap k=3. Idea 1 is now also the clean enabler of idea 2.
  - vLLM's own ReplaySSM/RecoverSSM (`config/cache.py:322-335`, `use_replayssm`) is the same
    idea, but only for Mamba2 and KDA (`supports_replayssm` is not on the Qwen3.5 classes).
    It's a design reference, not something we can switch on.
- **Cheapest test (ceiling, ~1 GPU-min):**
  - In a microbenchmark: a CUDA graph of [GDN spec kernel at 16 requests x 3 tokens, 24 heads →
    out_proj Marlin → gate_up Marlin], comparing the stock kernel against the kernel with the
    per-token stores of positions 0..k-1 removed (wrong outputs; timing only).
  - Expect ≥40 us saved per layer. Drop the idea if it is <20 us.

## Idea 2: Re-run 3 draft tokens with a KV budget that holds 16 requests (diagnosis F1)
- **Direction:**
  - Re-run T1-008's config (T1-009 flags, `num_speculative_tokens: 3`) with
    `--kv-cache-memory` raised from 5 GiB to the smallest value that holds 16 requests:
    16 x 365 MB ≈ 5.45 GiB, plus margin for prompts over 1600 tokens, so try 5.75 GiB, then
    6 GiB.
  - Check the server log for `Running: 16 reqs, Waiting: 0`.
  - If idea 1's second version lands, the extra budget isn't needed: the freed spec pages pay
    for the 3rd token.
  - Not proposed: disabling prefix caching (mode "none" needs 1+k pages instead of 2+k+1). The
    benchmark's 0% prefix hit rate makes that look benchmark-tuned. If anyone wants it, it needs
    a ruling first.
- **Why (measured):**
  - T1-008: acceptance 2.356 → 2.715 (+15%) and decode TPOT p50 -6%, even with 14 running.
  - F3: the target's Marlin goes from M=48 to M=64 for only +1.56 ms.
- **Upside (napkin):**
  - Extra step cost at 16 requests: Marlin +1.6 ms (measured, M1) + one more draft pass with
    FP8 head and local argmax ~1.3 ms (assumed) + a 4th GDN state write ~1.3 ms (from F2) +
    ~0.3 ms other. That is ≈ +4.5 ms on ~35.5 ms (+13%).
  - Against +15% accepted tokens per step, that nets **+2-4%**. Below the noise bar alone.
  - **With idea 1: +5-8% on top of idea 1** (the 4th state write and the extra memory disappear,
    so the step grows only ~+3 ms ≈ 8%).
  - Confidence: medium on the diagnosis (the logs are unambiguous), low-medium on the net
    number.
- **Memory risk (to check):**
  - Measured: 22.89 GiB free at start, model 14.99, graphs 1.34, KV 5.0 → ~1.5 GiB not
    accounted for, minus the activation and sampler peak (unknown).
  - +0.75-1 GiB is plausible but not certain. Needs the 128K test and the full benchmark without
    OOM (task_definition: memory change).
- **Novelty:** T1-008 was dropped as "mechanism unknown". This is the mechanism, found in its own
  logs, so the result is a re-test with a cause, not a repeat.
- **Cheapest test (one job, ~12 GPU-min):** in the same job, T1-009 args vs k=3 with 6 GiB of KV
  (bench 400 each). Grep `Running:/Waiting:` and acceptance, then run `long_request 128000`.

## Idea 3 (lower; partner of idea 1): L2 prefetch of gate_up during the host-AR window (R2-C), re-scoped
- **Direction:** unchanged from R2-C. A side-stream read-only kernel touches the first ~35 MB of
  the next gate_up weights while the 8-SM host AR (43 us) runs.
- **What's new:**
  - F2 shows today's gate_up after GDN layers is slow because of dirty-L2 drain. A prefetch
    issued into that state would fight the write-backs, so do idea 1 first.
  - Isolated cold gate_up is 113.9 us at 783 GB/s (M1). M=3 runs at 891 GB/s, so the M=48 tile
    work is not the limit.
- **Upside:** if half the weight bytes hit L2, gate_up could drop by ~30-40 us x 64 layers ≈
  2-2.5 ms per step → **+3-5%**. Confidence low: whether Ada L2 keeps ~35 MB of streamed lines is
  unmeasured.
- **Cheapest test (~1 GPU-min):** in M1's harness, gate_up at M=48 cold vs after a 35 MB
  prefetch kernel of its own weights. Kill the idea if it gains <15 us.

## Ruled down with numbers
- **Marlin M=48 config retune (R2-D) as a direction:** the in-kernel penalty is only 1.6 ms per
  step (F3). The rest is interference that idea 1 removes. ≤ +2% realistic.
- **k=4:** the Marlin M=80 cliff, 17.2 → 24.7 ms per step (M1).
- **Reduced-vocab draft head (R1-J):**
  - The FP8 draft heads now cost ~2 x 0.7 ms (T1-007).
  - A 64K-token head saves ≤ ~1 ms per step (~2.8%).
  - A 96.7% token coverage (measured in r1) loses up to ~3.3% of draft positions, i.e. ≈ -3%
    acceptance.
  - Net ≈ 0, with a ruling-2 data-provenance burden.

## My bets
1. **Idea 1, the GDN replay.** It is the largest measured remaining gap in decode, ~2.6 ms of a
   ~35.5 ms step. It is bitwise-exact and has a 1-GPU-min ceiling test. It is also the enabler
   for idea 2.
2. **Idea 2, k=3 with enough KV for 16 requests.** It is nearly free to test (a flag and a
   budget), and it settles KB11. Run it early in the round as a side job, even if it only
   breaks even alone: its value is mostly multiplicative with idea 1.
3. Idea 3 only after idea 1 (same window, same bottleneck family).

Combination map:
- Ideas 1 and 2 stack super-additively (1 removes 2's added cost and memory).
- Idea 3 touches the same gate_up time that idea 1 partly fixes, so they don't add fully.
- All three are orthogonal to the comms work (host AR) and the logits diet already in the
  champion.
