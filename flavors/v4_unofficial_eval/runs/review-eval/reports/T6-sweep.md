# T6 sweep (leader-6): summary
Map: [1-inputs](T6-sweep/1-inputs.md), [2-graph](T6-sweep/2-graph.md), [3-calibration](T6-sweep/3-calibration.md),
[4-knobs](T6-sweep/4-knobs.md), [5-compare](T6-sweep/5-compare.md), [6-ranking](T6-sweep/6-ranking.md).
Measurement: nsys node trace of T3-007 (GPU job 98, agents/leader-6/prof1).
Main points: decode step ~25-27 ms (W8A8 GEMMs 16.1 ms at 76% of DRAM peak, HMAR 5.4, other GEMMs 4.5, GDN 2.4); mixed steps
~28-37% of wall, with NCCL AR ~45% of their kernel time (421 us/call vs bare 205 us: peer skew). My top phase-1 candidate (large-message
AR transport) had already been closed by T1. My top candidate now is a row-split GEMM->AR pipeline in mixed steps (taken from T5's #2 by agreement).
GDN DeltaLog stays with T5. k=3 is memory-blocked.
