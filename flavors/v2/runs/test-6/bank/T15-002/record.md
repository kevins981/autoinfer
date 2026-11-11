team:         T15
builds_on:    BASE (microbench only, no served change)
performance:  "microbench: PCIe duplex with node-0 pinned slots: copy engines ~full duplex, SM-driven writes are not; CE-write + SM-read near duplex"
summary:
  hypothesis: KB5 (T1-003, "SM up/down traffic only partly overlaps: 41 us vs 27") predates the NUMA fix (KB37/KB67)
              and might be a far-node artifact. If PCIe duplex is available, the decode AR (~41 us = write ~17-21 +
              read ~19-23, serialized) has a ~20 us transport floor.
  change:     duplex.cu / duplex2.cu (this folder): mmap + mbind(node 0) + cudaHostRegister buffers (numa_maps: all N0),
              process pinned to node-0 CPUs; back-to-back 200 reps per mode; SM write (st.global.cs.v4), SM read
              (ld.global.cv.v4), copy-engine D2H/H2D on separate streams; 1 GPU and both GPUs at once. Jobs 117, 118.
  results:    Observed (us per op, both GPUs active; identical with 1 GPU):
              | bytes  | SM-W | SM-R | SM-W+SM-R | CE-D2H | CE-H2D | CE-D2H+CE-H2D | SM-W+CE-H2D | CE-D2H+SM-R |
              | 384 KB | 17.3 | 19.0 | 25.0-25.8 | 15.8   | 17.1   | 19.0          | 24.9        | 20.6-20.8   |
              | 480 KB | 21.2 | 22.9 | 30.0-30.4 | 19.6   | 21.0   | 23.5          | 34.4        | 25.1-25.3   |
              | 1.5 MB | 62.2 | 65.9 | 86-94     | 60.9   | 64.1   | 71.4          | 111.8       |             |
              | 4 MB   | 161.7| 170.7| 208-244   | 160.3  | 168.2  | 186.6         | 301.4       |             |
              SM+SM grows worse with more CTAs (384 KB: 25.0 at 32 CTAs, 27.8 at 128).
  conclusion: Interpretation: KB5 holds for the SM-driven path with near-node slots (not a NUMA artifact): SM posted
              writes are what break duplex (SM-W + CE-H2D is as bad as SM-W + SM-R). Copy-engine writes leave room:
              CE-D2H + SM-R is within ~2 us of CE+CE. A decode AR built as CE write + SM read (pieces with
              cuStreamWriteValue32 flags, SM reader polling) has a transport floor of ~21 us (384 KB) / ~25 us
              (480 KB uncompressed) vs ~41 us today. Estimated in-server: ~31 us/AR after CE start + tail latency,
              ~-10 us raw, ~-6 us net of the lost L2-prefetch window (KB41/T12) = ~0.75 ms/decode step, ~+2% e2e.
              T9-006's all-CE AR paid ~35 us of per-op latency at this size; whether the hybrid avoids it is unmeasured.
validity:     measurement only.
