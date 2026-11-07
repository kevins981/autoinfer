team:         T2
performance:  "no GPU A/B: analysis only (null result: <0.3 ms/step reachable by exact glue fusion)"
summary:
  hypothesis: the 1.6 ms/step inter-graph GPU idle in the champion's spec-decode step can be cut by fusing eager glue launches.
  change:     none (code read + graph-level nsys analysis of T2-006 config). Builds on T2-006. Workspace leader-2.glue untouched.
  results:    Observed: tiny sampling/verify/prep kernels run back-to-back (~1 us gaps; CPU ahead). 6 python gaps (124-292 us)
              around the drafter passes and before the target graph = 1.24 ms of the 1.6 ms idle; they sit around the FlashInfer
              attention-metadata build (numpy cumsum/where, copy_to_gpu, seq_lens D2H with spec decode, plan()), once per
              target + drafter pass. Report: agents/leader-2.w2/REPORT_glue.md.
  conclusion: Interpretation: launch fusion would save only ~0.1-0.2 ms; the idle is CPU-side python in metadata building.
              Exact micro-optimizations est. 0.06-0.2 ms. The larger fix (capture glue + sampling + drafter passes into one
              graph, est. 1.0-1.3 ms/step, ~3-4%) is invasive. Not pursued this round.
validity:     n/a (no change).
