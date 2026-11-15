# T1 sweep: summary
Map of the search space for Qwen3.8-27B-FP8 on 2x RTX 4090, built with framework/:
1. [1-inputs.md](T1-sweep/1-inputs.md): model, hardware, workload, contract, BASE design
2. [2-graph.md](T1-sweep/2-graph.md): nodes, indices, cross-device edges
3. [3-calibration.md](T1-sweep/3-calibration.md): napkin vs measured (decode 38.5 ms, mixed step 146.8 ms; comm 2-4x napkin, Marlin 1.5x)
4. [4-knobs.md](T1-sweep/4-knobs.md): sweep, knob by knob
5. [5-compare.md](T1-sweep/5-compare.md): vs T2's map (AR skew/transfer split: mixed-step AR = ~300 us transfer + ~390 us cross-rank wait)
6. [6-ranking.md](T1-sweep/6-ranking.md): ranked candidates
Split agreed with T2: T1 = all-reduce transport (I-AR), distributed argmax (R-ARGMAX), MTP k=3 test; T2 = prefill graphs, draft lm_head, W8A8 prefill, GDN commit.
