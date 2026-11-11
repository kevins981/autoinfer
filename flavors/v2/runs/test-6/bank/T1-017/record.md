team:         T1
builds_on:    T1-016
performance:  "measurement only (test-only trace, default off); not for official eval"
summary:
  hypothesis: the decode AR (KB19 ~6.8 ms/step) is partly peer wait/skew (KB8), not transport.
  change:     test-only %globaltimer trace in host_staged_ar.cu/.py (VLLM_HOSTAR_TRACE=<prefix>, default off). leader-2.w9,
              ws leader-2.arsplit, job 58, agents/leader-2.w9/arsplit/ (split.py, split_out.txt, NOTES.md).
  results:    Observed (425 16-req decode steps, 137 AR calls/step, 128 layer ARs at M=48 = 491,520 B):
              rank 0: kernel 42.0 us med = write 20.8 + wait 0.0 (mean 0.3) + read/add 21.0 -> 5.4 ms/step layer ARs.
              rank 1: 48.0 us med = 21.0 + wait 6.5 (mean 7.4) + 20.5. Rank 0 arrives last on 98.2% of ARs (KB8), skew med ~3 us.
              Wait larger after out-proj AR (rank1 8.5) than MLP-down (4.8). ~23.6 GB/s each direction. Gaps between ARs ~160 us.
              Estimate: ~2.4 ms/step wait; measured ~0 on the critical rank.
  conclusion: Interpretation: AR is pure PCIe transport at its floor on the pacing rank; skew/wait is not a lever. KB19's 53.6 us
              included nsys overhead. Only lever: hide transport behind compute (N-chunked row-parallel GEMM + AR pipeline).
validity:     instrumentation only, default off. Not for official eval.
