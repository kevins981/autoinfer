team:         T27
builds_on:    T27-001 (workspace t27-bud; env T27_BUDGET_UNIFORM=1 added)
performance:  "unofficial n=1 split run (job 185, 100 prompts c=16): adaptive plumbing with uniform k=2 layout and no ranking 811.5 vs champion-config 887.2 tok/s (-8.5%)"
summary:
  hypothesis: the e2e penalty seen with the budget (T5 bud3, T27-001) comes from the adaptive plumbing itself, not from the ragged layouts.
  change:     T27_BUDGET_UNIFORM=1: every step takes the stock layout (first 2 drafts per request, uniform), no ranking kernel, GDN uniform shortcut on, but the full adaptive path otherwise (k=3 drafted, adaptive cudagraph mode, compact_batch/reallocate_drafts, varlen decode graphs). Arm off = T27_BUDGET=0 in the same job, same build.
  results:    Observed (n=1 each, same job, fresh servers):
      off (champion k=2):      887.2 tok/s | TTFT p50/p99 194/1111 | TPOT p50 16.01 | ITL mean/med 36.0/25.3 | acc 2.357 | decode M=48 step med 25.33 ms (n=321)
      uniform-2 via adaptive:  811.5 tok/s | 221/1006              | 17.67          | 39.8/26.1             | acc 2.367 | 26.15 ms (n=300)
  conclusion: Interpretation: with no ranking and a stock layout the penalty is still -8.5% e2e (bigger than the -2.6% of T27-001, so n=1 noise is large, probably +-3%). The decode step is +0.8 ms, about the cost of drafting the unused third token; TTFT +14% and ITL mean +10.5% show the rest is outside the decode step. So the e2e penalty is in the adaptive plumbing (mixed-step handling / host work / adaptive cudagraph mode), not in ragged layouts or ranking. Not isolated further.
validity:     diagnostic only, not a candidate.
