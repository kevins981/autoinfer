# T2 sweep summary (leader-2)
Map of the search space for Qwen3.8-27B-FP8 on 2x RTX 4090 (TP=2, no P2P), built with framework/.
1. [1-inputs.md](T2-sweep/1-inputs.md): model, hardware, workload, contract, BASE's design (MRv2, MTP k=2, Marlin, PyNCCL, mixed steps eager above 96 tokens).
2. [2-graph.md](T2-sweep/2-graph.md): nodes with indices, missing-index table, serial edges, combine tags, knob sites.
3. [3-calibration.md](T2-sweep/3-calibration.md): graph-level profile of BASE (GPU job 2): decode 62% / mixed steps 38% of time; decode napkin 36.0 vs 38.6 ms; mixed steps CPU-dispatch-bound; GDN kernel writes 3 states per layer.
4. [4-knobs.md](T2-sweep/4-knobs.md): sweep over all six knobs + link/kernel rows, with pruning proofs (PP, DBO, DP mixers dominated at M=48).
5. [5-compare.md](T2-sweep/5-compare.md): vs T1: T2 adds mixed-step graphs (W1), 3-write GDN state, draft CPU gaps; disagrees on T1's prefill re-read and AR-overlap prices.
6. [6-ranking.md](T2-sweep/6-ranking.md): W1 (8-10%), KR1 GDN 1-write state (~4%), PL1 L2 prefetch during AR (0-9%), SP1 3 drafts, W4, RW1, SM4.
Framework findings: host dispatch is a missing resource; deferred writes drain into later windows; the link runs at ~12 GB/s + 20 µs.
Updates: update-<n>.md in T2-sweep/.
Updates: update-1..5 (update-5 = closing). Write-up: [T2-writeup.md](T2-writeup.md). Final HEAD T2-018, 838.0 tok/s official (+31.8% over BASE).
