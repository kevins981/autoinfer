team:         T12
builds_on:    BASE (CPU simulation only, no served change)
performance:  "CPU sim: cross-request n-gram drafting adds +0.7% E as an oracle at k=2 and +2.7% even with free rows (direction killed)"
summary:
  hypothesis: A suffix/n-gram index built online from the server's own earlier outputs (cross-request, plus the current prompt) supplies draft tokens with no GPU draft step. It could then replace or extend MTP chains where it matches long, without the 0.8 ms per draft step that capped deeper speculation (KB63).
  change:     bank/T12-001/xreq.py (copy of agents/leader-12/sim/xreq.py). Data: T3's self-generated set (agents/leader-3.w1/data: 400 seqs, 142,889 positions with the stock MTP's per-depth match flags; 200 thinking on, 200 off). Sequences are processed in order. Before a sequence is served, every earlier prompt and response goes into n-gram maps (n = 8, 6, 4, 3; longest match wins; most recent occurrence), plus the current prompt.
  results:    Observed (E = tokens per step per sequence):
    - MTP k=2 (reference): E 2.235, 3.00 rows/step.
    - oracle per step, best of MTP and n-gram, n-gram capped at 2 tokens: E 2.251 (+0.7%).
    - same oracle with the n-gram allowed 6 tokens and its extra rows FREE (upper bound): E 2.296 (+2.7%).
    - policy (n-gram when match n >= 6 and >= 3 tokens, else MTP): E 2.257 at 3.11 rows/step; tokens per row 0.726 vs 0.745 (worse).
    - Same-request prompt lookup (leader-8's agents/leader-8/ng/sim.py, rerun): oracle2 E 2.256, ng_or_mtp2 2.180.
  conclusion: Interpretation: drafts that cost no GPU time don't help on this model and data. The stock MTP already predicts the recurring phrases, and long verbatim continuations across requests are rare. Even the free-row upper bound (+2.7% E, roughly +1.5-2% end to end) is below the bar. This extends leader-8's prompt-lookup kill (KB66 notes) to cross-request suffix caches (SuffixDecoding-style). Caveat: one 400-sequence set (the benchmark also has ~400). Agentic or code-heavy loads might differ, but this run doesn't measure them.
validity:     measurement only (CPU); no served change. The data is T3's self-generated set (allowed under the 2026-11-08 ruling). Benchmark data was not used.
