# T5 sweep (leader-5): summary
Map: [1-inputs](T5-sweep/1-inputs.md), [2-graph](T5-sweep/2-graph.md), [3-calibration](T5-sweep/3-calibration.md),
[4-knobs](T5-sweep/4-knobs.md), [5-compare](T5-sweep/5-compare.md), [6-ranking](T5-sweep/6-ranking.md).
Measurements: GPU jobs 77 (nsys graph) and 78 (nsys node) of T3-007, agents/leader-5.w1/prof_*.
Main points: decode step 28-31 ms (body GEMMs 15.3 ms at 0.8 of BW, hmar 6.2, GDN 3.7, drafter 2.85, lm_head 1.6); mixed steps
~37% of time. My host-bound reading of mixed steps was an nsys artifact (T4-001 corrected it). Top candidate: GDN DeltaLog with a
row-0 checkpoint layout (+3-4% if the state writes reach DRAM; ncu pending); second: row-split GEMM->AR overlap in mixed steps.
Work split with T4: T5 = DeltaLog; T4 = AR+norm fusion, k=3.
