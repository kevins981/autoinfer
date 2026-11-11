team:         T48
builds_on:    BASE (standalone microbench, no engine change)
performance:  "measurement only: no engine change; hardware compression gives no gain on real model data"
summary:
  hypothesis: The RTX 4090 may support transparent, lossless hardware "generic compression" of device memory
              (cuMemCreate with CU_MEM_ALLOCATION_COMP_GENERIC). If real FP8 weights, bf16 lm_head or similar data
              compressed under it, decode weight streaming would get faster with zero SM cost. This matters because
              software codecs lost here: decode instructions cost more than the bytes they save once weights are
              L2-warm (KB102).
  change:     comp*.cu (in this record): plain vs COMP_GENERIC allocations, 256-512 MB buffers (> 72 MB L2), filled from the host
              with synthetic patterns and real checkpoint bytes (layers 10-11 MLP FP8, the first 512 MB of the bf16 lm_head);
              read bandwidth with an ld.global.cs streaming kernel, 256 MB flush before each rep, best of 7. GPU 0.
              Jobs 419, 421, 424 (1.3 GPU-min total).
  results:    Observed:
    - CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 1; compressible allocation granted.
    - Plain allocation: 953 GB/s for every pattern (no compression).
    - Compressible allocation, effective read GB/s (job 421/424):
      zeros 8192-8738 | constant byte 3197 | 16 B pattern repeated in 128 B 3161 | 32 B pattern in 256 B 946 |
      random bytes 953 | real FP8 weights 953 | real bf16 lm_head 953 | FP8 exponent bits in place (byte&0x78) 947 |
      FP8 exponent as low nibble, 1 per byte 1208 | 2 exponents per byte 946 |
      bytes with 1 or 2 random low bits 1859 | 3 or 4 random low bits 949 | 100+rand(0..3) 1859 | 100+rand(0..7) 947 |
      u16 with 4 or 6 random low bits 1859 | u32 with 8 or 12 random low bits 1859 | fp32 with random high 16 / zero low 16 946 |
      fp32 in [1.0,1.1) 947 | half of 128 B blocks zero 1638 | half of 32 B sectors zero 1139 | 1 random byte in 4 or 8 1859-1916.
    - CPU check on real bytes: 0 of the 128 B blocks are all-zero or have a byte range <= 3, in both the FP8 MLP and the lm_head
      sample; 0.000% zero bytes in FP8, 0.195% in bf16.
  conclusion: Interpretation:
    - The 4090 has working, transparent compute-data compression. Its ratios are quantized: 1x, ~2x, ~3.4x (constant or
      16 B-periodic blocks), ~8.6x (zero blocks), at roughly 128 B block granularity.
    - The 2x mode behaves like base + fixed narrow delta per element: up to ~2 of 8, 6 of 16 or 12 of 32 bits. It is not
      entropy-based: real FP8 exponent bits (~32% of their byte's bits as information) don't compress.
    - So for non-constant data the best case stores x bits of information in >= ~2.7x bits and halves them: worse than
      dense packing. No exact re-layout of high-entropy data (FP8 weights at ~6.5 bits/byte, bf16 lm_head, fp32 GDN
      state, FP8 KV) can read fewer than 1.0x its packed bytes through this path.
    - It pays only for data with all-zero or constant 128 B blocks. This workload has none of material size: the
      weights have none, and new GDN states start at zero but are never read before they are written.
    - Door closed for this task. A usable fact for workloads with block sparsity (zero tiles read at ~8.6x for free).
    Caveats: GPU 0 only; L2 behaviour of compressed lines and write-path cost not measured; the delta-model reading is
    inferred from ~30 patterns, not documented.
    Relation to T49-002 (job 418, one job earlier, found independently): same headline (supported; zeros ~8.7x,
    constant ~3x, real data 1x). This record adds the compression rule (quantized ratios; base+narrow-delta criterion,
    tested on ~30 patterns) and the argument that it can't beat dense packing. Difference: T49 saw -4..-5% read
    bandwidth on incompressible data in compressible memory (1 GiB buffers); I saw no penalty (953 vs 953 GB/s, 512 MB,
    job 419). Unresolved; both agree: don't put model data in compressible memory.
validity:     microbench only; no served-path change.
