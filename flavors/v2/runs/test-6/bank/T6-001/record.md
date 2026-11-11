team:         T6
builds_on:    T1-042
performance:  n/a (measurement record; no serving change)
summary:
  hypothesis: FP8 Marlin re-streams the weights per 64-row slab, so M=65-127 costs ~2 weight passes; M=49-64 has a config step.
  change:     none (microbench agents/leader-6.w1/mb1.py, adapted from T5-002's mb_marlin.py; job 31, 0.7 GPU-min; ws t6-gemm built).
  results:    Observed (GPU0, cold L2-flushed, per-step sum over 48 GDN + 16 attn layers, per-rank TP=2 shapes):
    Marlin ms: M=3 14.34, 48 15.54, 52-64 17.63-17.67, 72 25.43, 80 25.80, 96 28.05, 112 29.65, 128 25.87, 160 35.76, 192 35.39, 224 45.08, 256 46.57.
    Roofs: HBM read 955 GB/s (12.16 GB -> 12.74 ms); bf16 4096^3 133 TF (GPU0), hot M=128/256 real shape 106/120 TF.
    M=72 gate_up = M=64 time + ~88 us remainder launch. GPU1 ~8-10% faster, same shape.
  conclusion: Interpretation: marlin_mm splits M into 64-row launches; each launch streams all weights (gate_up/down > L2).
    Two full slabs in one launch share L2 (M=128 < M=72-112). Headroom vs max(mem,compute): 25-45% at M=72-112, ~22% at 49-64, ~0-20% at >=128.
    Not covered by KB44/T1-032 (that merges the remainder launch but keeps serial 64-row slab passes; not in the champion).
validity:     measurement only.
