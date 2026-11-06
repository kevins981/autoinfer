# Scoring (novelty/upside 1-5)
- Prefetch L2 during AR (both proposers, #1): nov 5, up 3 -> SELECTED (core)
- Cheaper AR (both proposers, #2; NCCL env / PCIE_IPC / custom): nov 3, up 4 -> SELECTED (first step, cheap, same idle-DRAM window)
- PP=2: nov 5, up 4 but proposed risky; systematic engineer argues it doubles weight bytes per request -> not selected (archive)
- Marlin tuning+SiLU fusion: nov 2, up 2 -> archive (candidate round 2)
- k=3 / max-num-seqs: nov 2, up 3, not FFN -> archive
- N-split overlap: nov 3, up 1 -> archive
