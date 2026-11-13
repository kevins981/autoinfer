# T55 write-up (research-handoff, leader-55). Recommendation: CLOSE. GPU used: 0 of 30.

Triage: reports/T55-triage.md. Sparring: T55-1 (sparring/). Split with T54 (reports/T54-triage.md): T55 took ideas 1 and 5.

## The memo, idea by idea
- **Idea 1 (DeltaLog Keep).** Predicted: removing per-step state writes cuts GDN HBM traffic ~43% (151 -> 86 MB/req/step).
  Found (from the run's records, not re-measured): this system already does it. The champion's replay kernel reads the
  checkpoint plus a token log and writes no state per decode step (KB15, KB48, KB111). The memo's premise of a write
  every step was right for stock vLLM and wrong for this engine. The remaining cost is the dense fp32 state read
  (25 MB/layer/GPU, 38.5-40 us vs ~25 us floor). Sparring confirmed the read is required (dense per-step q, shared across
  verify rows already, no cross-request sharing, no room in L2). Headroom ~0.7 ms/step, <=2% e2e, and T22 found that
  L2 prefetch in the AR window doesn't reach the warm time. Broken assumptions: 16-bit state (it is fp32) and writes on
  every step.
- **Idea 5 (shared-prefix lazy state).** Predicted: 5-15x less state traffic for the first ~150 tokens of branched
  requests. Found: the benchmark has no fan-out. Its only shared prefix is the chat template, and prefix-state caching
  already gave +0.5% (KB86). The crossover is ~190 tokens at fp32/TP=2, below typical context lengths. Broken
  assumption: the workload.
- **Ideas 2, 3, 4, 6, 7** (T54's slice, my verdicts in the triage): the memo's ridge (~330 rows) and PCIe AR latency
  (10-15 us P2P) are wrong here. The exact ridge is ~87-180 rows, there is no P2P, and an AR costs ~50 us. So "extra
  verify rows are free" (idea 2) fails at c=16. Ideas 3/4 target long contexts the benchmark doesn't have. Idea 6.2 and
  idea 7A were already built (KB112/123, KB84). Idea 7B is approximate.

## Lesson about the memo
Its first-principles method is sound, but three hardware/model inputs are off: state dtype, exact-arithmetic ridge, and
P2P/AR latency. Its target regimes (B=1-4, long context, fan-out) aren't this benchmark. Its best idea (replay instead of
writing state) had already been found and shipped by this run.
