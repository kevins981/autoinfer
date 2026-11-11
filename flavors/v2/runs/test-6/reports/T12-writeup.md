# T12 write-up (leader-12, explore). Recommending a voluntary close after the seed, with no proposal

GPU: 0 of 20 used. Record: bank/T12-001 (CPU only). Notes: local_kb/leader-12.md. No worker started.

## Measured kill: drafting with no GPU cost, from a cross-request n-gram cache (T12-001)
- **Idea:** take draft tokens from an online suffix index over the server's own earlier outputs plus the prompt (SuffixDecoding-style).
  - These drafts need no 0.8 ms GPU draft step, which is what capped deeper speculation (KB63).
  - So they could extend chains cheaply where matches are long.
- **Simulation:** T3's self-generated data (400 sequences, the stock MTP's per-depth match flags), sequences processed in order.

| policy | E | rows/step |
|---|---|---|
| MTP k=2 | 2.235 | 3.00 |
| best of MTP / n-gram per step (oracle), 2 tokens | 2.251 (+0.7%) | 3.00 |
| same oracle, 6 tokens, extra rows free (upper bound) | 2.296 (+2.7%) | 3.00 |
| n-gram when match >= 6, else MTP | 2.257 | 3.11 (fewer tokens per row) |

- **Interpretation:** the MTP head already predicts the recurring phrases. Long verbatim repeats across requests are rare. This extends leader-8's prompt-lookup kill to cross-request caches.

## Costed without GPU time (all below the bar)
1. **Speculation economics:** a cheaper drafter combined with budgeted K_MAX=3 at M=48, reusing T5's built two-slot GDN, confidence kernel and adaptive verification.
   - Measured inputs (T5-003/006): k=2 step 25.25 ms; budgeted K3 step 26.21 ms; E 2.357 -> 2.489. A draft step costs lm_head 310 + MTP Marlin 200-263 + fc 80 + AR 110-183 us.
   - Assumed savings per draft step, about -0.33 ms:
     - 32K draft vocab (allowed by ruling 2), lm_head ~110 us;
     - FP8 fc (would need a ruling), ~40 us;
     - drafter AR down to ~60 us.
   - Also assumed: exact top-1 screening of the target lm_head, -0.3 ms per step (T10's idea).
   - Result: step 24.94 ms, E 2.477, decode throughput x1.064. Closed-loop step model: the trial drops 96.3 -> ~92.3 s, so **payoff if it works ~+4.3%**. Below the bar on its own.
   - K_MAX=4 gives about the same, because its extra E is cancelled by one more draft step.
2. **Full-duplex copy-engine AR (decode and small mixed steps):**
   - KB55's 4-piece DMA AR moves 2 x 3.93 MB per GPU in 255 us, at least 30.8 GB/s combined.
   - The decode hostar's byte part already runs at about 26 GB/s combined (2 x 383 KB in ~29 us).
   - A perfect duplex at about 46 GB/s would save about 12 us per AR. About 40% of that comes back as lost L2 prefetch window (KB41).
   - Net about 7 us/AR, about 0.9 ms/step, **about +2.5% end to end**. Mixed steps of 256 tokens and up belong to T9 already.
3. **Uneven 32+16-row decode microbatches** (no m-tile padding) to hide the ARs:
   - Cost: the B passes add about +36 us per layer on gate_up plus about +20 us on the other GEMMs.
   - Gain: the hidden ARs, minus the lost prefetch credit, are worth about 48 us per layer.
   - Net about 0. The tensor and HBM coupling (KB69) kills it, just as it killed the even split.
4. **L2-prefetching the GDN state in the AR window instead of weights:** this just trades bytes inside the window. The latency it saves is about 0-10 us/layer, at most ~1%.
5. **Exact INT8/INT4 slicing:** at least 9 int8 or 30 int4 MMAs per bf16 MMA, so it is slower (agrees with T2/T10/T11).
   - Writing a dense matrix as a sum of two 2:4-sparse matrices just breaks even.
   - Native 2:4 groups are rare in FP8 weights.
6. **Data-parallel replicated drafter (no ARs):**
   - Each GPU reads the whole drafter per step: about 0.62 ms against ~0.5 ms for TP.
   - It costs 0.25-0.5 GB more memory per rank.
   - Result: no gain.

## Leftover worth an exploit-type owner (not a direction)
- **Anomaly:** the drafter's 2 ARs cost 110-183 us per draft step (T5-003). Host AR at M=16-37 should take ~50 us.
  - Cause unknown: skew, the logits gather, or a non-hostar path.
  - Worth ~0.2-0.3 ms/step (~1%) at k=2, more with deeper drafting.
- **Package:** the costed exact items, all 0.5-3% each:
  - item 1 above;
  - T2-006's mixed-step list;
  - T10's admission-lag and lm_head items;
  - T9's rank rebalance.

  Stacked, they might give +6-9%. But KB45 is a warning: an earlier stack of small items measured null.

## Opinion
- I agree with T2, T4, T8, T10 and T11 that the champion is pinned by coupled floors:
  - bf16 tensor time per row: exact FP8 or integer routes don't exist on sm_89;
  - FP8 weight bytes per step;
  - PCIe bytes per AR. KB55 caps duplex gains at about 7 us/AR net.
- Drafts are capped by MTP quality. Neither fine-tuning (KB60) nor free n-gram sources (T12-001) move it.
- I found no direction with an honest payoff of +5% or more. I'm returning the 20 GPU-min seed.

## Addendum: T12-1 (stacked package) funded, then killed at G1 (bank/T12-002, jobs 105-106, 10.5 GPU-min)
- **Plan:** make each draft step cheaper (32K draft vocab, replicated FP8 fc), then spend the savings on budgeted K3.
- **Results, job 106** (48 prompts, one fresh server per config):

| config | draft step | acceptance | tok/s |
|---|---|---|---|
| T1-042 | 0.835 ms | 2.430 | 801.7 |
| frequency-ranked 32K + FP8 fc | 0.574 ms | 2.229 (-8.3%) | 743.1 (-7.3%) |
| frequency-ranked 32K, bf16 fc | 0.583 ms | 2.258 | 751.8 |

- The draft-step gate was met, but the throughput gate failed.
- **Vocab coverage:** a 32K list misses about 8% of the target's benchmark-time tokens, frequency-ranked or not.
  - Held-out coverage on T3's data: 91.6% frequency-ranked vs 87.0% for the first 32K ids.
  - T3's self-generated data is narrower than benchmark text.
- **FP8 fc:** costs about 1.3% acceptance for about 10 us per step. Not worth it.
- **A2 finding:** the drafter's "2 ARs" are 4 host collectives: embed AR, fc all-gather, attention AR, MLP AR+norm.
  - Draft step 1: 171 us. Draft step 2: 112 us.
  - The attention and MLP ARs are normal. The excess is embed-AR skew plus the fc gather.
- **Interpretation:** the draft lm_head can't shrink without losing more acceptance than the time it saves. KB21 says 98K costs nothing, so 98K is already about the right size.
  - Without cheap draft steps, K3 falls back to T5's ~+1.7%.
  - The package's payoff if it works falls below +3%, under the bar.
- **Reusable:** `draft_lm_head_token_file`, a generic id-list remap.
