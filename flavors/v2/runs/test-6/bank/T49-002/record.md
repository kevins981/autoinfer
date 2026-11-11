team:         T49
builds_on:    BASE (microbenchmark only; no engine change)
performance:  "microbench: hardware compression gives 0 on our data (-4..-5% read BW)"
summary:
  hypothesis: Ada (sm_89) supports compute-data compression (cuMemCreate with CU_MEM_ALLOCATION_COMP_GENERIC). If it
              compresses any of the decode step's dominant bytes (FP8 weights 12.2 GB/step, GDN fp32 state ~1.2 GB/step),
              it is a free, exact bandwidth gain with no ALU cost (unlike the software codecs of KB102). Never tried in the run.
  change:     none. bank/T49-002/cmp.cu: 1 GiB buffer (> L2) in normal vs compressible VMM allocations, streaming
              16-B __ldcs read kernel, best of 7 reps, GPU0. Data: zeros, constant byte, real FP8 weights of layer 10
              (gate/up/down/qkv/z/out_proj, 383 MB tiled), the high-nibble and low-nibble planes of those weights
              (entropy 5.50 / 7.96 bits/byte; whole byte 6.55), random bytes, fp32 N(0,.01), smooth fp32 AR(1) series, bf16 N(0,1).
  results:    Observed (job 418, 0.4 GPU-min): GENERIC_COMPRESSION_SUPPORTED=1, compressible allocation granted.
              Normal allocation: 956-957 GB/s for every pattern. Compressible: zeros 8322 GB/s (8.7x), constant byte
              2767 GB/s (2.9x); FP8 weights 921, hi-nibble plane 925, lo-nibble plane 926, random 926, fp32 911,
              smooth fp32 909, bf16 909 GB/s (-3.7..-5.0%).
  conclusion: Interpretation: the 4090's hardware compressor is real and large, but only for zero/constant tiles; data
              with any per-sector variety (even a 5.5-bit plane, or smooth fp32) does not compress and pays ~4-5% for
              compression metadata. Weights, GDN state, KV and activations here are all of that kind, so this lever is
              closed; do not allocate model data in compressible memory. Possible residual use only for data that is
              literally zero in bulk (none found on the critical path).
validity:     n/a (no engine change)
