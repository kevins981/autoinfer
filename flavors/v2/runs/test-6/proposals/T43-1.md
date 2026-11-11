# T43-1 (research, leader-43): Don't communicate rows that can't matter: row-selective all-reduce and the target's "decision depth"

## The observation this rests on (from the run's own numbers)
- A decode row costs ~0.10 ms (T37-001 step model: 16.9 ms + 0.10 ms/row + 0.83 ms/draft step, jobs 368/369/372).
  About 60% of that is PCIe all-reduce bytes, not compute:
  - all-reduce time is linear in rows with ~0 intercept, ~0.85 us per row per AR (KB32/KB41; T37-001: 48 -> 104 rows takes ARs 5.9 -> 12 ms);
  - 128 ARs/step -> ~0.06 ms/row; the GEMM row term is ~0.04 ms/row.
- At k=3, 1.27 of every request's 4 rows are discarded after the forward (acceptance 0.774/0.552/0.400 cumulative, job 369): ~20 of 64 rows per step, ~2 ms/step (~8% of the step), of which ~1.2 ms is PCIe bytes for rows whose result nobody reads.
- The run's view of speculation depth is "AR/drafter-bound" (KB95: k=3->4 capped at ~+1-2%). The coupling that sets that cap is that *every* draft row pays full communication through all 64 layers.

## The idea (what is new here)
A row's communication can be dropped at any layer without touching exactness, as long as a request drops only a *suffix* of its rows and its accepted length is capped at its last kept row:
- kept rows' sums are bit-identical (all-reduce, norms and GEMMs are per-row; attention and GDN are causal within the request, so kept rows never read dropped rows);
- a dropped row only means "this draft was not proposed", and verification of everything kept stays exact.

So the decision "is this draft row still worth paying for?" can be made late and cheaply, and its first effect is on **bytes on the link**, which is where a row's cost actually is on this machine.

Mechanism: a **row-selective all-reduce**. The host-AR kernel reads a device-side live-row mask and packs and sends only live rows. Graph shapes, the uniform k-layout and every other kernel stay unchanged; dead rows carry finite garbage that is never read.
- This avoids the ragged-batch plumbing that sank budgeted-K (KB85: -8.5% from plumbing alone, with a uniform layout and no ranking).
- GEMM savings (row compaction) are an optional later step; the AR savings need none.

Where the mask comes from, in rising cost:
1. **Before the forward:** the drafter's chained confidence (KB60: confidence ranks acceptance well). This is budgeted-K's signal with none of its plumbing.
2. **Mid-forward ("decision depth"):** at layer L, a cheap probe on row j-1's hidden state (draft token's lm_head row · h_L, or T27's PCA screen) predicts whether draft d_j will be rejected; if so, drop rows j.. for layers > L.
3. **Structural:** at the last layer, non-final prompt rows of a prefill need neither the out-projection AR nor the MLP (state and KV come from the layer's input). Small (~1/64 of prefill ARs), but free and exact.

## Relation to prior work (honest)
- Early rejection inside verification exists in the literature: HiSpec (arXiv 2510.01336, early-exit verifier layer) and FASER (arXiv 2604.20503, in-pass pruning of rejected tokens in vLLM, compute-side).
- Budgeted/adaptive K (T5, T27, T36) and confidence ranking (T3) exist in this run.
- New here:
  - **(a)** moving the pruning into the communication layer under fixed graphs, which is what matters on a no-P2P TP stack where a row's marginal cost is mostly PCIe bytes;
  - **(b)** a measurement nobody has: the *decision depth* of this hybrid GDN/attention model, i.e. at which layer the target's verdict on a draft becomes predictable, for accepted vs rejected drafts;
  - **(c)** the reframing that the speculation-depth cap (KB95) is a cap on *communicated* rows, not on rows. If (b) shows early decisions, deeper k (k=4-6, two-slot GDN already makes state memory k-independent) gets re-opened with a different cost per row: full cost for rows that survive, partial cost for rows that die.

## What it could teach future teams, whatever the outcome
- The decision-depth curve: fraction of eventually-rejected drafts identifiable at layers 16/32/48 at a given false-drop rate, and how that differs between GDN and attention layers.
  - If decisions are late (only the last few layers), mid-forward pruning of any kind (compute or comm) is dead on this model; that is a firm negative for HiSpec/FASER-style ideas here.
  - If they are early, deeper speculation needs re-costing.
- A measured exchange rate between dropped rows and step time with a row-selective AR (is AR time really linear in live rows inside the chain, with L2-prefetch coupling (KB78) taking some back?).

## Napkin payoff (k=3, today's champion), to be replaced by measurement
- Pre-forward confidence masking alone: if it drops ~25% of doomed rows at <1% acceptance loss: 5 rows x 0.06 ms = 0.3 ms/step (~1%).
- Mid-forward at L=32 with 70% recall: 0.5 x 0.7 x 20 rows x 0.06 ms = ~0.45 ms/step (~1.7%); with GEMM compaction ~0.75 ms (~3%).
- Upside case: at k=5-6, doomed rows are ~50% of rows. If decisions are early (L~16) the depth cap moves, maybe +5-8% e2e. That needs the probe to be good, and is the reason to measure before building anything.
- Odds of a >=+5% outcome: low (~10-15%). Odds of a clear, reusable answer to "is mid-forward pruning viable on this model": high.

## First step (inside the 20 GPU-min seed)
1. **(GPU, ~8 min) One dump job on the champion (T37-002 workspace, k=3, ~64 ShareGPT-free prompts).**
   - Use self-generated / non-benchmark prompts, per the trained-artifact rule.
   - Hook verification rows: the hidden state of every row at layers 8, 16, 24, 32, 40, 48, 56 (bf16, ~10 KB/row/layer, ~200 MB total).
   - Also record: the draft tokens, the final target argmax per row, the accepted length and the drafter confidences.
2. **(CPU) Decision-depth analysis.**
   - Logit lens (final norm + lm_head on the CPU or in the dump job) and a 1-parameter probe (draft-token logit margin vs top-1 of the approximate logits from T27's PCA screen).
   - Output: ROC of "will be rejected" per layer; recall at 0.5% / 1% false-drop; a combined pre-forward (confidence) + mid-forward model of rows saved; a predicted ms/step at k=3..6 using T37's step model.
3. **(GPU, ~5 min) AR microbench in the champion's 16-layer chain.**
   - Host-AR with 64 vs 48 vs 32 live rows of 64 (packing by mask).
   - Checks that AR time follows live rows in situ, including the prefetch coupling (KB78).
4. **(~4 min) Reserve.**

Gate for asking more (stage 2: build the row-selective AR + confidence mask, ABBA at k=3 and k=4/5, ~40 GPU-min):
- predicted gain >= +2% at the best k; and
- mid-forward recall >= 50% at <= 1% false drops by layer 32, or confidence masking alone predicting >= +1.5%.
Otherwise I file the decision-depth curve and the AR-vs-live-rows measurement as the result and stop.

## GPU time
Seed 20 GPU-min covers the first step. No new grant requested now.
