team:         T1
builds_on:    T1-017
performance:  "microbench only: N-chunked GEMM + AR pipeline projects ~0.35 ms/step (~1%) at best -> NO-GO"
summary:
  hypothesis: splitting row-parallel GEMMs (down, out-proj) along N and running chunk i's host AR on a 2nd stream while chunk i+1
              computes hides ~2 ms/step of AR transport (T1-017).
  change:     microbench only (agents/leader-2.w9/ovl/: ovl.py, job.sh, r1/, NOTES.md; job 59, 0.2 GPU-min), real host AR + champion
              Marlin fp8, CUDA graphs, weights cycled to stream from HBM.
  results:    Observed (rank 0, us/layer-call): down M=48 serial c=1 111.2; pipelined c=2 105.6 (-5.6), c=4 127.2 (+16);
              oproj M=48 c=1 77.5, c=2 83.3, c=4 102.6. AR runs concurrently with Marlin (P recovers 70-75% of overlappable time),
              but each extra GEMM chunk costs ~10 us (Marlin fixed per-call) and each extra AR ~9-12 us. Chunked GEMM != bitwise
              (1 ulp, k-split order). Estimate ~2 ms/step; measured best ~0.35 ms/step.
  conclusion: Interpretation: overlap is physically possible (AR uses 8 SMs + PCIe, Marlin leaves room) but chunking overheads eat it.
              Overlap must come without splitting the GEMM/AR.
validity:     microbench, no server change.
