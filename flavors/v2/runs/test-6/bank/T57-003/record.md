team:         T57
builds_on:    T52-005 (analysis only, CPU; T56's nsys job 483)
performance:  "analysis: drafter section 1.90 ms of a 24.77 ms decode step; ~0.76 ms/step above byte floor; fusible exact items ~255-310 us/step upper, ~130-155 realistic"
summary:
  hypothesis: the drafter has ~1.4 ms/step of non-byte overhead (KB126, T56 trace) worth fusing.
  change:     none; per-kernel anatomy of 50 decode-only steps, both ranks (agents/leader-57.w1/task3/REPORT.md, kernel_tables.txt).
  results:    Observed: section 1898 us/step = verify glue 58, level0 (64 rows) 665, level1 541, level2 533, tail 101; 148 kernels; idle gaps 72 us (52 in eager glue). Per level 1/2: GEMM 285, comm 108 (4 hostar 102 + AG 6), screen/sampling 91, attention 21, norms/emb 15, elementwise 11. Level 0 comm 192 (ARs at 64 rows). MTP GEMMs at byte floor (200 vs 199 us); fc 69 vs 55; screen kernels +31 us above floor per level. Embedding ARs 83 us/step.
  conclusion: Interpretation: KB126's ~0.5 ms/level non-byte overhead is not supported (215 us/level at levels 1-2, 0.34 ms at level 0, ~0.76 ms/step total). Fusion list (worker report) ~1.0-1.25% e2e ceiling. Biggest piece is ARs (~0.41 ms/step), reducible by removing the drafter's embedding AR (T56's precomputed fc-embedding table) and compacting level 0.
validity:     analysis only.
