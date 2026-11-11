team:         T1
builds_on:    T1-030
performance:  "microbench only: split chain -1.4%/layer at M=300 (172/128), +0.7% at 448, -6.3% at 512; below the 3% bar -> dropped"
summary:
  hypothesis: compressing the T1-025 DMA lane (chunk 0) with T1-030's codec shortens the split chain further.
  change:     dma_staged_ar.cu/.py: encode kernel to a device staging buffer (fixed-size planes+headers through D2H/H2D; overflow raw
              blocks zero-copy into the host slot), fused decode+add after H2D; opt-out VLLM_DMA_COMP=0. ws leader-2.arcomp2.
              Worker leader-2.w18 (agents/leader-2.w18/comp/dchain.py, d1/; job 125, 0.3 GPU-min).
  results:    Observed: bitwise 0 mismatches (300 real + 30 adversarial). DMA AR alone raw -> comp: M=172 168.4 -> 145.9, M=300
              272.3 -> 228.8 us (0.84; encode/decode kernels ~15-20 us). Chain us/layer: M=300 1142.6 -> 1126.9 (-1.4%), M=448
              1524.3 -> 1535.6 (+0.7%), M=512 1804.9 -> 1691.0 (-6.3%).
  conclusion: Interpretation: chunk 1's compute already hides most of the DMA AR; the encode kernel steals SMs from the concurrent
              GEMM. Expected <= +0.3% end-to-end. Not pursued.
validity:     exact (bitwise), not evaluated.
