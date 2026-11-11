team:         T44
builds_on:    T37-002 (+ T44-002's knobs; test-only labelled cudaEvents in dma_staged_ar.cu, off unless trace_start)
performance:  n/a (decomposition; job 403, 0.1 GPU-min)
summary:
  hypothesis: The two-lane copy-engine AR of split mixed steps (VLLM_T9_PINGPONG=1) runs well above its correctness-required
              critical path (T44-002: 247 us at 300 rows vs ~170 bound); the 20-35 us per-op CE start latency other teams
              measured need not accumulate on that path.
  change:     per-piece event timeline of DmaStagedAllReduce (4 pieces, two lanes, both ranks) at 300 and 700 rows;
              cross-rank flag latency as a pair average (globaltimers offset ~76 ms). agents/leader-44.w1/s1/out3/.
  results:    Observed: tracing adds ~17-20% (300 rows: 289 -> 338 us; 700: 468 -> 563). Untraced varies between jobs
              (300: 247 vs 289). Rank 0 lane 0 at 300 rows (traced): D2H 112 us busy out of the 211 us between the first D2H
              start and the last D2H end; D2H end -> flag written 102 us total over the pieces (13-60 us each); observe -> H2D
              start (the F reset) 79; H2D 109; add 7.5-12. Flag latency ~2 us for piece 0, 10-35 for pieces 1-3. 700 rows: D2H
              221, D2H end -> flag 176, reset 42, H2D 196, ack before join 24.5.
  conclusion: SUPERSEDED by T44-004 (intervention showed no gain: the D2H end event marks issue, not landing).
              Original interpretation: the gap is code-imposed ordering, not CE hardware latency. (1) cuStreamWriteValue32 doorbell
              on the D2H stream completes behind in-flight posted writes and blocks the next piece's D2H (~60% of the gap);
              (2) the F reset sits before each H2D (~15%); (3) the ack is written before the join (~5%, more at 700 rows);
              (4) CE start gaps ~8%; tail ~10%. With only the required dependencies the estimate is ~190-200 us traced at
              300 rows (untraced ~165-185 vs 247-289: ~-30-35%) and ~-15-20% at 700 rows. It matches the
              bytes-at-duplex-rate + one-piece-latency bound (~180 / ~380). Fix (not built): a separate doorbell stream
              per lane waiting on an event after each D2H; resets moved after the last H2D, before the ack; join recorded
              before the ack. Same bytes, same add: bitwise. Risk: visibility of the D2H data before the flag on the
              cross-stream path; needs a KB57-style negative control.
validity:     measurement only; instrumentation off by default.
