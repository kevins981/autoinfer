team:         T3
builds_on:    T3-001
performance:  n/a (offline simulation; no serving change)
summary:
  hypothesis: The stock MTP head's confidence ranks draft acceptance well enough that verifying only the batch's best B draft slots (out of K_MAX=3-4 per request) yields more accepted tokens per unit of step time than fixed k=2.
  change:     agents/leader-3.w1/dump_conf.py (K=4 chains, confidence = softmax max over the 98,304-id draft vocab, match vs target greedy), sim_budget.py (20,000 batches of 16).
  results:    Observed (job t3-conf5, 0.3 GPU-min): 142,889 positions; cum acceptance .765/.541/.378/.268; fixed k=2 E 2.316 (in-server 2.325). Confidence >= 0.95 on 47% of step-1 drafts, accepted 97.6%; mid bins overconfident. Step-time model 25.5 ms at M=48 + 0.28 ms/token + 0.55 ms/draft step: E/t vs k=2: fixed k=3 0.973; K_MAX=3 B=32 1.061; K_MAX=4 B=32 1.067; upstream rule (d) K_MAX=4 1.070; sensitivity 1.053-1.093. Ranking by stale (previous-position) confidences: ~1.01.
  conclusion: Interpretation: about +6-7% decode-step throughput if slot selection uses current-step confidences (on-device) and the extra draft steps cost about 0.55 ms; a CPU budget from stale confidences loses most of it. Basis for proposals/T3-2.md.
note:         the confidence dump (job conf5) ran in the unbuilt t3-distill.dump workspace (KB61); the MTP/target math is the same (replay is bitwise), so acceptance/confidence data are unaffected; only step-time inputs from T3-007 are in question.
validity:     No serving change; data provenance as T3-001.
