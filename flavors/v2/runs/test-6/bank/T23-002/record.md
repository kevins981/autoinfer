team:         T23
builds_on:    T14-004 (estimate only; ws t23-mb built from T14-004, unmodified)
performance:  "n/a: estimate, gate G2a failed"
summary:
  hypothesis: skipping dead m-blocks (rows past M) in the last Marlin slab saves >=1.5 ms per small mixed step, exactly (bitwise on live rows).
  change:     none built. Estimate from in-situ data (agents/leader-23.w1/W1-2/REPORT.md).
  results:    Observed: under T6's slab policy, dead m-blocks average 0.74 per small mixed step, 0.97 over all mixed steps (job-134 counts); slope ~22 us per m-block per layer (3-39 us) from W1-1's per-layer Marlin sums.
  conclusion: Interpretation: <= ~1.0 ms per small step (<=1.85 at the steepest slope), below the 1.5 ms gate; the B-dequant dominates a slab and is unaffected. Not built. Note (W1-3): Marlin actually sees cudagraph-padded M, so this estimate on pt+dt is if anything high.
validity:     n/a.
