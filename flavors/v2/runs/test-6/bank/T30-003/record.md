team:         T30
builds_on:    T29-006
performance:  "unofficial: VLLM_DMA_PIPE=8 within noise, 16 worse (pt209-400 +14 ms, pt401-1000 +7 ms)"
summary:
  hypothesis: the copy-engine AR's fill/drain (~0.4 of ~0.8 ms per big-step AR, T30-002) shrinks with finer pieces.
  change:     NP cap 4 -> 16 in dma_staged_ar.cu/.py (bitwise identical, same bytes and add). Sweep VLLM_DMA_PIPE=4/8/16. By leader-30.w1.
  results:    Observed: job 264, 120 prompts per arm, probe. Median ms at pt 209-400 / 401-1000: P4 93.9/139.6, P8 96.8/139.7,
              P16 107.8/146.5. pt>=209 fit: 0.158+40.4, 0.152+44.2, 0.144+56.6. Decode is the same in all arms; acceptance
              is the same. H2D is already flagged per piece (code read).
  conclusion: Interpretation: per-piece overhead (~0.2 ms/layer-AR at 16) cancels the shorter fill. Both chunks' ARs share
              one link at the same moment, so finer pieces can't help. Staggering (T30-1) is the lever. Not pursued.
validity:     exact (bitwise by construction); not proposed for official.
