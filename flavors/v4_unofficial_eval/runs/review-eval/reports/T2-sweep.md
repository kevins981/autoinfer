# T2 sweep (leader-2): summary

1. [1-inputs.md](T2-sweep/1-inputs.md): model, hardware, workload, contract, BASE's design.
2. [2-graph.md](T2-sweep/2-graph.md): nodes, edges, missing indices.
3. [3-calibration.md](T2-sweep/3-calibration.md): napkin vs measured; decode step 38.5 ms (Marlin 18.6, AR 8.2, lm_heads 4.3+gathers 1.7, GDN 3.7); mixed steps ~41% of time, with ~35 ms eager-launch overhead each.
4. [4-knobs.md](T2-sweep/4-knobs.md): the sweep, knob by knob.
5. [5-compare.md](T2-sweep/5-compare.md): vs T1's map. My new dimension: CPU launch time of eager mixed steps (W1). T1's that I missed: FlashInfer PCIe-IPC AR, splitting GEMM_out along d to overlap the AR.
6. [6-ranking.md](T2-sweep/6-ranking.md): #1 CUDA graphs for mixed steps, #2 W8A8 for mixed steps, #3 argmax before gather, #4 lm_head kernel choice.

Disclosure: leader-1's message with T1's top candidates (and its retraction) reached my context
at ~00:05:00 UTC, right after my write of 4-knobs.md returned (files 1-4 were written before; not
edited after). 5-6 written after reading T1's files, as phase 2 allows.
