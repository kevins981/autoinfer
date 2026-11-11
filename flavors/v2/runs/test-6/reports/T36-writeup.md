# T36-1 (moonshot): fill every weight pass — speculation depth set by confidence, load and the 16-row tile, with state memory decoupled from depth

## What I measured on the champion T31-003 (all my own jobs; unofficial)
| Observed | Source |
|---|---|
| GPU 0 busy 97-98% of wall time (15.23 s of 15.67 s). No host slack. | job 356, graph trace |
| A decode-only step costs 18.6 ms at M<=12 rows and ~+1.6 ms per extra 16-row tile (23.4 ms at M=48): **~80% of a decode pass does not depend on the number of rows.** | job 356, period vs rows |
| Inside one decode step: body 21.6 ms (GEMMs ~12.1, 128 ARs x 43 us = 5.5, GDN/attn/norm ~2.5), target lm_head 1.0, one MTP draft step 0.55-0.62 ms, prep 0.35. | job 356, node trace |
| Mixed (prefill) step: GPU busy = 30.8 + 0.160 ms x prompt tokens. Decode-only steps 72.5% of time, prefill extra 20%. | job 356, fit over 46 steps |
| MTP acceptance per position at K=3: 0.770 / 0.542 / 0.388, i.e. conditional 0.77, 0.70, 0.72: **the chain barely decays with depth.** | jobs 361, 363 |
| K=3 at c=16 ties K=2 (1015.3 vs 1015.7) only because **2 of 16 requests wait**: "Running 14, Waiting 2, KV usage 95.3%". TPOT is 10.5% better, TTFT mean 541 vs 207 ms. | job 361 |
| Raising the KV budget to 5.8 GiB to fit them: CUDA OOM at graph capture. Both GPUs sit at 23.8 GiB at K=2. | job 362 |
| Same-job K=3 vs K=2 at c=14 (where K=3 fits): 993.9 vs 973.5 tok/s, **+2.1%, one pair, 150 prompts** (pair sigma ~0.7-1%). Step +10.1% (25.03 vs 22.73 ms), tokens/step +15.9%. | job 363 |
| K=4 does not start (the GDN replay registry asserts). | job 361 |

## The idea, and the assumption it rejects
The champion treats speculation as one fixed choice: **K=2 for every sequence, in every step, at every load,
with K+2 recurrent-state blocks reserved per request and group.** That choice is pinned by memory, not by
economics: the state allocator scales with K, the GPUs are full, so K=3 already evicts two requests.

But a decode pass is ~80% fixed cost (the weights must be streamed once per pass whatever M is), and the
drafter's conditional acceptance stays ~0.72 at depth 3. So the right question per step is "which rows are
worth 0.1 ms each", not "what is K". I would build:
1. **State memory independent of depth.** The champion's own checkpoint+replay kernel uses one committed state
   S[0] plus a log S[1]; the other speculative state blocks are only touched by the rare flush path. Allocate 2
   (+checkpoint) blocks per request and group at any depth, redirecting the flush. This frees memory at K=2 and
   makes any depth fit 16 requests.
2. **Rows chosen by confidence under a tile budget.** Draft up to D=4-6 deep, then keep the rows with the highest
   estimated survival probability up to a budget that lands on a 16-row tile (64 or 80 rows). The engine already
   has this machinery for another draft method (`adaptive_verification.py`: confidence -> budget -> variable-length
   verification, cost curves); it is not wired for MTP, and it needs a confidence from the screened draft head
   because `use_local_argmax_reduction` never materialises draft logits.
3. **Depth by load.** With n<=5 running requests the extra rows are free (same tile), only the 0.6 ms draft step
   costs: K=6 there. (`num_speculative_tokens_per_batch_size` exists in the scheduler.) A generic low-load policy
   driven by the measured cost curve, not by the benchmark's concurrency.
Verification stays exact and full-vocab against the target: output = the target's greedy output.

## Upside
Cost model from the measurements above: step(rows, depth) = 23.4 ms at (48, 2), +1.6 ms per tile, +0.6 ms per
draft step. Acceptance model: p1 0.775, then 0.72 per depth (measured to depth 3, **assumed** beyond), with
per-position heterogeneity **assumed** (Beta-distributed accept probabilities; simulation in
`agents/leader-36/`, run in this session).
- Static K=3 / K=4: +7.0% / +9.0% on decode steps (model). Measured for K=3: +5.3% on decode steps, +2.1%
  overall. The model overstates by roughly 1.4-2x at the "overall" level; I scale the lines below accordingly.
- Budgeted depth, D=5, 64-80 rows: +11.6-12.5% on decode steps with a well-calibrated confidence, +8-10% with a
  noisy one (model).
- **If it fully works: ~+8-9% over the champion** (decode gain x ~0.7 decode share, plus ~1% from deep drafts
  in the low-load tail, where 3-4% of seat-time sits). It clears the 5% bar, not by a wide margin.

## Riskiest assumption and cheapest test (= milestone 1)
**That the drafter's own confidence separates accepted from rejected rows well enough for budgeting to beat
static K=3.** If it does not, the direction caps at static K=3/4, about +2-4%, below the bar.
Test: at K=3, c=14 (fits today, no memory work), log per row (depth, screened-head top-1 probability / margin,
accepted?). One ~3 min job. Offline: AUC per depth, then replay my simulator on the real joint data for budgets
48/64/80. Pass: predicted decode gain >= 9% (>= ~5% overall after the scaling above). Fail: I stop and write
up.

## Roadmap
1. Calibration log + offline replay (above). ~8 GPU-min including one repeat.
2. 2-block state allocation: K=3 with 16 running requests in 5 GiB; bitwise check of the state path one request
   at a time; 128K request; ABBA vs champion. Must show: 16 running, TTFT back to ~champion, gain >= the c=14
   pair. ~20 GPU-min.
3. K=4-6 enabled (replay registry/log cap, CUDA graph sizes), budgeted rows via adaptive verification for MTP.
   ABBA vs stage 2. Must show a further consistent gain. ~35 GPU-min.
4. Load-based depth; GSM8K + 128K; official eval. ~35 GPU-min.

## Honest forecast
Central estimate **+4-5%** official over the champion; odds of >= 5%: **~35%**; odds of any consistent gain
(>= 2%): ~65%. Measured: p1-p3, the cost per tile and per draft step, the memory block, +2.1% for static K=3
at c=14 (one pair). Assumed: p4+ (flat conditional), the heterogeneity/calibration of confidence (milestone 1
measures it), that 2 state blocks suffice outside the flush path (read from the champion's code, not tested),
that adaptive verification can be wired to MTP without losing the fused GDN replay path.

## What I looked at and dropped (so nobody repeats the napkin work)
- Pipeline-parallel / sublayer split / single-GPU with compressed weights: every scheme other than TP reads
  the weights twice per sequence-step, or uses one GPU's DRAM only. Dead by byte arithmetic (not measured).
- Prefill chunks riding in decode steps: each mixed step pays ~7.7 ms fixed (GDN chunk path), so more, smaller
  mixed steps lose (job 356 fit).
- A concurrent weight-prefetch stream: after a sparring session the usable DRAM-idle windows outside the ARs
  are ~40 us/layer; <= ~9% with two unmeasured steps. Disclosure: `gpu wait` on a wrong job id showed me the
  tail of job 357 (T35's, a cold/warm GEMM microbenchmark). I did not use it, and I leave that area to them.

## GPU time
Balance 12.1 of the 30 seed (17.9 used: jobs 356, 360-363; job 360 was lost to my own script bug, 2.0 min).
Milestone 1 fits in the remaining seed. Stage 2: +20. Whole roadmap: ~100 GPU-min.
