# T3 sweep: summary

Framework: framework/ (six knobs over the serving graph). Champion when written: T2-008 (880.1 tok/s).

1. [1-inputs.md](T3-sweep/1-inputs.md): model, 2x RTX 4090 TP=2 with no P2P (all-reduces through pinned host memory), workload, contract, champion design.
2. [2-graph.md](T3-sweep/2-graph.md): data and compute nodes per layer, their indices and missing indices.
3. [3-calibration.md](T3-sweep/3-calibration.md): napkin vs measured decode step (26.9 vs 33.3 ms on T1-014). The gap is mostly GEMM efficiency (body GEMMs ~0.7 TB/s in situ) and the AR. The link floor is 22.6 GB/s for D2H+H2D at once, so the decode AR is at ~87% of its floor.
4. [4-knobs.md](T3-sweep/4-knobs.md): the sweep. The machine-specific framework number is that each verify row costs ~0.12 ms of link time per step, which kills k=3 and trees.
5. [5-compare.md](T3-sweep/5-compare.md): against T1/T2. My DeltaLog candidate was built independently by T2 (T2-008). New from my sweep: the link floor, the per-row link price, and NCCL running 2.7x slower in the server than isolated.
6. [6-ranking.md](T3-sweep/6-ranking.md): (1) AR-window L2 work (prefetch the next GEMM's weights plus drain the GDN state's dirty lines during the PCIe-bound AR), (2) copy-engine AR for mixed-step row-split, (3) DeltaLog L=2, (4) reduced-vocab draft head with a generic frequency table.

Tested so far: T3-001, hostmem AR for all sizes, was a screen negative (-6.6%, job 79). Fp16 accumulation is dead (fp16 overflow). Candidate 1 is in progress (worker leader-3.w1).
