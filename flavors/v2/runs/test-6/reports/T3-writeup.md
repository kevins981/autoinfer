# T3 write-up (leader-3, explore). Closed voluntarily after the last chance

GPU: about 39.5 of 120 granted used (about 80.5 returned). Records: bank/T3-001..T3-008. KB60 (plus KB61's
lesson, which we hit too). Worker: leader-3.w1 (all implementation and measurement). Champion
reference throughout: T1-042, 942.2 tok/s.

## Lever chosen
Tokens per decode step (E). T2's anatomy (T2-006) and ours agree that every part of the decode step is
within about 10% of a measured floor, so E multiplies everything. Champion: E = 2.325, per-position
acceptance 0.77 / 0.55 cumulative (T1-042 official logs), at k=2.

## T3-1: self-distilled MTP draft head (payoff if it works +10%). Closed on its own kill criterion
- **Pipeline (T3-001).**
  - Data: target-generated prompts (387 self-instruct from a hand-written generic topic list, 13 Magpie;
    Magpie yields mostly fragments on this model) and T=0 replies: 144K response tokens, 200 thinking
    on / 200 off.
  - Post-norm hidden states were dumped from rank 0. The split is 360/40 by prompt.
  - An offline PyTorch MTP (vLLM-faithful recursion) reproduces in-server acceptance: held-out
    0.781/0.561 vs 0.77/0.55.
  - Provenance: agents/leader-3.w1/data/PROVENANCE.md.
- **Fine-tunes. Nothing moves held-out acceptance:**
  - T3-002: full fine-tune on hard labels made it worse (lr 1e-5: a2 −1.3 pt; 5e-5 overfits).
  - T3-003: restricted (dedicated step-2 layer, LoRA, lr 2e-6, masked step-2 loss) gave ±0.25 pt,
    CI includes 0.
  - T3-004: KL to the target's own logits, computed free from stored hidden states via the target
    lm_head. Train a1/a2 +5/+12 pt, held-out flat (+0.13 pt), held-out KL −11/−17% with no argmax change.
    The data slope is about +0.35 pt a2 per doubling, so 15x more data extrapolates to about +0.6%
    throughput.
- **Conclusion (KB60):** the jointly trained stock head is near what one layer can learn from data we
  can afford to generate (about 45K tokens per GPU-min of generation). Don't spend GPU on head training
  unless a much cheaper data source or a bigger draft architecture is on the table.

## T3-2: confidence-budgeted verification (payoff if it works ~+6%, measured down to ~+3%). Stopped
- **Idea.** Draft K_MAX = 3 steps per request. Verify only the batch's best B = 2 × num_reqs slots
  (M = 3 × num_reqs, i.e. the champion's width at any load), ranked on GPU by survival = cumprod of
  draft confidence. Exact: verification is unchanged.
- **Offline (T3-005/006, held-out, no training).** Confidence ranks acceptance well: 47% of step-1
  drafts have p >= 0.95 and are accepted 97.6%; mid bins are overconfident; isotonic calibration helps
  slightly. E gain at fixed M=48: +8.1-8.4% (K_MAX=3), +11% (K_MAX=4). Ranking must use the CURRENT step's
  confidences (stale ranking: about 0); a CPU-chosen B from stale confidences is fine.
  Admitted slots per request are ragged, 0..K_MAX.
- **In-server costs (T3-008, built workspace; T3-007's timings were from an unbuilt workspace, KB61):**
  - forward 22.4 ms at M=48 (21.9 at M=42), sample/reject 1.12, propose 1.69, i.e. 0.84 ms per draft step;
  - in-server k=3 acceptance matches the offline chain (.77/.56/.41 vs .765/.541/.378);
  - an apparent verify cliff just above M=48 was seen only unbuilt; it's unmeasured on a built engine.
- **Net:** K_MAX=3, B=32 ≈ +4.6-4.9% decode-step (sim), ≈ +3.8% after an assumed 0.2 ms selection/varlen/confidence overhead, ≈ +3-3.5% end-to-end (85% of wall time scales).
  K_MAX=4 is worse, because the extra draft step costs more than it adds. Sim with built costs:
  K_MAX=3 B=32 raw 1.046, iso-calibrated 1.049 (E 2.539 vs 2.348, step 26.04 vs 25.20 ms); K_MAX=4 B=32 1.040/1.045; B>32 below 1 under the assumed (unbuilt-measured) cliff shape (bank/T3-008/notes/).
- **Blockers to land it.** Each GDN group allocates 1+k state slots per request, so blocks/request =
  1 + 3(1+k): 10 at k=2, 13 at k=3. At k=3 only 14 of 16 requests fit in 5 GiB (the KB13 stall).
  Option A, a bigger pool, is dead: 16 requests need >= 5.62 GiB, and 5.65/5.70/5.75 GiB OOM in
  CUDA-graph capture. The remaining routes:
  - A2: drop part of the prefill-graph grid (frees about 0.6 GiB; the every-32 grid was worth about +1%, KB34);
  - B: rework T1's replay so that flush (gdn_replay_flush modes 0/1) and the align-mode copies
    (mamba_utils.py) use one materialized slot instead of S[0..a-1]. The replay kernel's no-log
    fallback reads S[accepted−1]. Then num_speculative_blocks = 1 for any k. That's exact (same values,
    different slot), but every reader must be found; test with the 128K request (it crosses 1600-token
    boundaries). Bonus: it frees about 1.3 GB/rank of KV at k=2, though that has no direct speed value
    at c=16.
- **Code to reuse.**
  - Upstream `v1/worker/gpu/spec_decode/adaptive_verification.py` (DSpark only): ranking kernel, batch
    compaction, varlen graph route; gdn_attn.py already has a non-uniform-spec branch.
  - Needed changes:
    - allow method=mtp and lift the greedy-ids exclusion;
    - get the confidence from a sharded sum-exp folded into the local-argmax all-gather;
    - B = 2 × num_reqs.
  - T1's replay supports ragged lengths up to 3 drafts (kMaxLogTokens = 4).
  - Plan: agents/leader-3.w1/PLAN_T3-2.md. Estimated 3-4 days of code, riskiest in the ragged FULL-graph
    target path.
- **Verdict:** about +2-3% official after the memory-fix cost. A reasonable exploit item if someone does
  route B; not an explore direction.

## Where tokens-per-step headroom might remain (opinion)
- **The ceiling is real but hard to reach.** Offline K=4 cumulative acceptance is .765/.541/.378/.268.
  More E needs either more verified tokens (each costs about 0.15-0.2 ms below M=48, more above) or a
  better drafter (KB60: not by fine-tuning on affordable data).
- **Cheaper draft steps** would make K_MAX=3-4 pay. Each step is 0.84 ms against roughly 0.5 ms of
  weight bytes: MTP layer FP8 about 0.18 GB/rank, the bf16 fc about 52 MB/rank, the FP8 98K draft lm_head
  about 0.25 GB/rank, plus 2 ARs. Fusing the drafter's K steps into one graph without host round trips,
  and keeping the small MTP weights L2-resident across steps (they are about 0.43 GB/rank, which doesn't
  fit; the fc and norms would), are untested.
- **Verification width per request shaped by confidence** is the right structure. Combined with route B
  above and a measured built-engine cliff map, it's worth +2-3%.
- **Process lessons:**
  - check `_C` is the built one before any timing (KB61); it cost us a wrong resync forecast;
  - CUDA-graph capture, not the KV allocation, is the binding memory constraint near 5.6 GiB.
