team:         T1
builds_on:    T1-004
performance:  "microbench negative: overlapped GEMM+AR slower than serial (+8..14 us at c=2), T1 GPU job 26"
summary:
  hypothesis: (framework Window knob, W1) split the row-parallel GEMM's output columns into c chunks (disjoint weight
              columns -> no re-read), all-reduce chunk k on a side stream while chunk k+1 computes; napkin hides ~25 us of
              each ~43 us HOSTMEM AR -> ~3.4 ms/decode step (~+4%).
  change:     standalone microbench only (agents/leader-1.w1/ov/): real Marlin FP8 + HOSTMEM AR, 2 ranks, CUDA graph.
  results:    Observed (us, graph, K=8704/M=42): serial GEMM+AR 79.1; chunked serial c=2 94.0, c=4 134.5; overlapped c=2 91.6,
              c=4 127.0. Independent GEMM || AR on two streams (no dependency): 76.2 vs 81 summed -> only ~5 us concurrency
              (K=3072: ~11 us). Chunking costs: GEMM sum 1.2x (c=2) / 2x (c=4) of full; AR fixed cost ~12-15 us per call.
              (Isolated GEMM times are L2-assisted: 44.6 MB weights fit the 72 MB L2 in a loop.)
  conclusion: Interpretation: the framework's Window cost max(F, BW, Link) assumes the two nodes can share the device; here
              Marlin occupies every SM (persistent blocks) and the AR kernel's spinning blocks need SMs too, so the "link"
              node is not on an orthogonal resource — its PCIe traffic is issued by SM loads/stores. Plus per-chunk fixed
              costs. Dropped. A copy-engine AR (cudaMemcpyAsync D2H/H2D, no SMs) is the only form in which the link is truly
              orthogonal; its latency per call is likely higher.
validity:     n/a (not built).
