team:         T46
builds_on:    T37-002 (measurement only; no change to the served path)
performance:  "microbench only: host zero-copy embedding gather 30.8 us at 64 rows vs vocab-parallel lookup + AR (~45 us with real hostar, KB109)"
summary:
  hypothesis: the bf16 embedding (1.27 GB/GPU, vocab-parallel, one AR per target pass and per draft level) can live in
              pinned host memory, each GPU gathering its rows over PCIe, exactly and no slower than lookup + AR.
  change:     none served. Bench in agents/leader-46.w1/hm/ (RESULTS.md), job 422, GPU0, CUDA graphs, medians of 24-30 reps.
  results:    Observed: (a) host zero-copy gather (full 248320x5120 bf16 table, cudaHostAlloc mapped, node 0) 30.8 us at
              64 rows (21 GB/s, link-bound), 11.2 us at 16 rows; output bitwise equal to the device rows. (b) device
              lookup 4.1 us + single-GPU AR stand-in (655 KB host write + read, no peer wait) 55.3 us at 64 rows;
              16 rows: 18.4 us total. Estimate was 25-35 us for (a).
              Not measured: two GPUs gathering at once; gather beside a concurrent AR; in-server.
  conclusion: Interpretation: exact; frees ~1.27 GB/GPU; vs the real hostar (~43-45 us, KB109) saves ~13-15 us per
              embed AR, ~4 per step at k=3 -> ~0.05 ms/step (~0.2%). Its value is the memory (enabler for memory-blocked
              items, e.g. T42's screening plane), not speed. Hand-off item for an exploit team.
validity:     measurement only. A served version is bitwise (same rows); the vocab-parallel AR it removes sums one
              non-zero partial with zeros, so outputs are identical.
