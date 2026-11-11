team:         T1
builds_on:    T1-020
performance:  "microbench gate only: hybrid DMA/hostar 2-way token split -13.3% per layer at M=300 (172/128), -12.5% at M=320; pure DMA split -7.5%"
summary:
  hypothesis: prefill-size ARs (~0.875 us/token/AR, ~11.7 s/trial) can hide behind GEMMs if their transport runs on the copy engines
              (prefill Marlin fills every SM, KB33) and the per-token chain is split into 2 token chunks.
  change:     2-process microbench only (agents/leader-2.w12/dmaar/r1..r3, NOTES.md; jobs 86, 88, 92; 4.5 GPU-min): DMA AR = D2H to
              pinned shm slot, cuStreamWriteValue32 doorbell, cuStreamWaitValue32, H2D, add kernel; CUDA-graph capturable.
  results:    Observed: DMA AR bitwise == hostar at all sizes; alone 12-20 us/AR slower than hostar (M=300 333 vs 318). In the layer
              chain (out_proj->AR->norm->MLP->AR->norm->in_proj), same job vs serial hostar: M=300 split 172/128 + DMA -7.8%, 150/150
              +14.9% (both halves pay Marlin's remainder launch); M=320 192/128 -6.0%; M=150 86/64 -2.1%. Hybrid (chunk0 DMA, last
              chunk hostar): M=300 -13.3% (1656 vs 1910 us/layer), M=320 -12.6%; high-priority stream hurts (-8.5%, -11.1%).
              DMA stamps in chain (M=300): D2H 260, doorbell->release 8 (rank 0) / 71 (rank 1), H2D 256, add 3-4 us: copies at ~12 GB/s
              in-chain vs 21-24 GB/s alone (unexplained). Token split changes Marlin reduction order: max diff ~0.07 (fp reorder).
              Estimate (napkin) exposed AR 532 -> ~130 us/layer at M=300; measured split hides ~35% (pure DMA).
  conclusion: Interpretation: hybrid passes the -12%/layer gate; projected ~2-3 s/trial (+2-3%) for mixed steps >= ~256 tokens.
              In-server build approved (w12, ws leader-2.dmaar2 from T1-023).
validity:     microbench only.
