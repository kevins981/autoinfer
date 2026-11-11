team:         T46
builds_on:    T37-002 (measurement only)
performance:  "microbench only: host-fed weight slice. SM zero-copy (8 CTAs) at f<=2.5%: chain -1.0..-1.2% (stand-in consumer); real t31 on host-mapped B: +28% (loss); copy-engine staging ring: loses at every f"
summary:
  hypothesis: PCIe is idle ~75% of a decode pass; feeding a fraction f of a decode GEMM's FP8 weights from pinned host
              memory beside HBM adds ~22-25 GB/s (~2.5%) of weight bandwidth; a copy-engine ring consumed from L2
              might do it without DRAM traffic.
  change:     none served. agents/leader-46.w1/hm/ (RESULTS.md, scripts, raw data); jobs 420 (ncu), 422, 425, 426. GPU0,
              M=64, real t31 down_proj (N=5120 K=8704, 44.6 MB) for the HBM part, READ flush + persisting-L2 reset per rep.
  results:    Observed: single GEMM f=0 56.3 us. Host part runs ~23 GB/s; break-even f ~2.4% (as estimated).
              (a) SM zero-copy, 8 CTAs, xor-reduce stand-in consumer: f=1.47%/2.5% 55.3 us = ideal (HBM part alone);
              f=5% 93.2. 32 CTAs slow the GEMM (65.7-220 us).
              Chain 4x[AR stand-in -> GEMM] f=0 426.0 us: (a) 420.9 (f=1.47%) / 421.7 (2.5%), -1.0..-1.2%, = ideal;
              (b) CE ring 443.7-537.6, worse at every f.
              ncu: CE-staged 1.1 MB consumed with 87% L2 hits; 4.46 MB 100% L2 misses (full DRAM read); no later dirty
              write-back of CE lines (vs 4.5 MB after an SM-written control).
              Real t31 with host-mapped B slice: bitwise equal to HBM slice, but pulls host data at 6-12 GB/s: f=2.5%
              93-184 us; chain +28%.
              Caveat: the chain runs AR and GEMM in sequence (no link contention); real hostar carries L2 prefetch blocks.
  conclusion: Interpretation: the link IS additive bandwidth beside an HBM-bound GEMM (not zero-sum) when driven by a
              few SM CTAs with many outstanding 16 B loads, but its size caps the gain at ~2.4% of weight-stream time
              (~0.3 ms/pass, ~1% e2e) and only if the GEMM gets a dedicated host-slice loader (t31's own pipeline is
              latency-bound on host memory). Copy-engine writes go through to DRAM (inferred: no dirty write-back,
              misses above a few MB), so a "PCIe -> L2 channel that skips DRAM" does not exist here. Not pursued
              further: ceiling ~1% against a non-trivial kernel build.
validity:     measurement only; host-fed slices are bitwise (same bytes, same kernel arithmetic).
