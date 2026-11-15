# T1 sweep summary (leader-1)
Map of the search space for Qwen3.8-27B-FP8 on 2x RTX 4090 (TP=2, no P2P), from the framework in framework/.
1. [1-inputs.md](T1-sweep/1-inputs.md): model, hardware, workload, contract, BASE's design.
2. [2-graph.md](T1-sweep/2-graph.md): data/compute nodes, indices, edges, missing-index observations.
3. [3-calibration.md](T1-sweep/3-calibration.md): napkin vs measured per part; the model reproduces the official 143.2 s
   (decode steps ~58%, prefill-containing steps ~42%; communication ~29%, Marlin ~48%).
4. [4-knobs.md](T1-sweep/4-knobs.md): the sweep over Keep/Rewrite/Speculate/Split/Place/Window with napkin gains and pruning reasons.
5. [5-compare.md](T1-sweep/5-compare.md): vs T2's partial map (no history yet): T2's Marlin-re-read observation and L2-prefetch-during-AR site added.
6. [6-ranking.md](T1-sweep/6-ranking.md): top: faster all-reduce (~15%), prefill AR/GEMM overlap, L2 prefetch during AR, read-once prefill GEMM, decode GEMM, draft head FP8/local argmax.
Updates: update-<n>.md in T1-sweep/.
