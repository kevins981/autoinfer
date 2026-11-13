team:         T53
builds_on:    T50-002 (standalone microbench on the champion's t31b kernel; no engine change)
performance:  "measurement only: chunk-major t31c GEMM passes gate A on down (C2 1.054x, C4 1.081x); first posting attempt fails gate B"
summary:
  hypothesis: The row-parallel GEMM that feeds a decode AR (down M=64 N=5120 K=8704; gdn out_proj K=3072) can finish
              its output in column chunks spread over the kernel (gate A: <=1.10x stock, even spread). Each chunk
              can then be posted to pinned host memory while the GEMM keeps computing, taking the AR's write half
              off the critical path (gate B: GEMM slowdown <=+5 us).
  change:     agents/leader-53.w1/ext/t53c.cu ("t31c", built via ext/build.py, build dir named by source hash).
              It reuses t31b's inner loop unchanged, with persistent CTAs walking items (chunk, z, slab, kseg) in
              chunk order. Split-K: the last arriver reduces in a fixed segment order. Each chunk has a completion
              counter. Weights are prefetched across items (v2; v1 is kept as ext/t53c_v1.cu). Posting: post1 =
              the tile-finishing CTA writes host + fence.sys + seq flag; post2 = dedicated poster CTAs.
              Bench: benchAB.py / jobAB.sh. Cold = 256 MB read flush + cudaCtxResetPersistingL2Cache. GPU0,
              20 reps round-robin, CUDA events with the GPU kept busy during enqueue, %globaltimer per chunk.
              Host buffers verified on NUMA N0. Jobs 441 (Python crash), 442 (v1), 445 (v2): 1.1 GPU-min.
  results:    Observed (medians):
              - down: stock 57.3 us. No chunking (C1, persistent + staged epilogue) 60.4 (1.054x).
                C2 UN2 kseg1: 60.4 (1.054x), chunks done at 29.7 / 56.3 us.
                C4 UN2 kseg3 P120: 62.0 (1.081x), chunks at 18.4 / 32.8 / 47.1 / 58.4. C8 1.36x.
              - out_proj: stock 24.5. C2 26.6 (1.085x). C4 1.38x. C8 >= 2.2x.
              - v1 (no cross-item prefetch): every chunked config >= 1.14x.
              - Posting, down C4 kseg2 P80: 8 poster CTAs (256 threads, 4x16 B) slow the GEMM +7 us (62.5 -> 69.6).
                The last chunk is posted 10.3 us after GEMM done, i.e. ~16 GB/s.
                post1 (GEMM CTAs write): GEMM +25 us. out_proj: posting +8..+13 us.
              - Numerics: kseg=1 configs are bitwise equal to stock. kseg>1: max-abs 6e-5 on outputs up to
                0.038 (down). Host copies and flags verified.
  conclusion: Interpretation: chunk-major scheduling works for down at 2-4 chunks, and for out_proj at 2 chunks
              only. Items shorter than ~4 K-blocks per warp group cost too much. Posting as first implemented
              fails gate B (see T53-002 for the second attempt). Details: agents/leader-53.w1/RESULT.md.
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
