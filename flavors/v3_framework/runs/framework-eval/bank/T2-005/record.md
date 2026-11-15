team:         T2
builds_on:    T1-004 (HOSTMEM AR) + T2-003 (Triton GEMM); microbenchmark only, no engine change
performance:  "microbench negative: 0-5.5 us saved per AR+GEMM pair (~1.4% of a decode step at best), AR +2-4 us"
summary:
  hypothesis: (framework Window knob, link || DRAM) while the host-memory AR runs (32 blocks, DRAM idle), a prefetch
              kernel on ~96 free SMs loads the next GEMM's weights into the 72 MB L2; napkin ~30 us saved per AR
              (~3.8 ms/decode step).
  change:     standalone microbench (worker leader-2.w1: agents/leader-2.w1/bench3.py, ext/pf.cu): 2 ranks, CUDA graph,
              per layer HOSTMEM AR [45x5120] then GEMM at M=45 (Triton W8A8 or Marlin), >=2.7 GB of rotating weights;
              prefetch on a forked stream (prefetch.global.L2, L2::evict_last, or ld.global.cg), first X MB.
  results:    Observed (T2 GPU job 33): serial vs best prefetch, us/layer: in_proj Triton 93.4 -> 88.9; qkv 86.4 -> 83.0;
              gate_up 148.7 -> 143.4; down 93.3 -> 92.6; Marlin similar (-3.6 to -5.5). AR+prefetch alone 45.5-48.3 vs AR
              44-46 us. Napkin ~30 us vs measured 0-5.5 us (10-15% of napkin).
  conclusion: Interpretation: the prefetched bytes mostly don't turn into L2 hits for the GEMM (mechanism untested:
              access order vs Triton tile walk, eviction, or the GEMM at M=45 not purely DRAM-bound). Not worth building.
              Open check if revisited: ncu L2 hit rate of the GEMM after prefetch.
validity:     n/a (no engine change).
