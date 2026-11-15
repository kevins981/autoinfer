team:         T3
builds_on:    T3-009 + T3-004
performance:  screen negative (1 pair, -2.45%) (ABBA job 130)
summary:
  hypothesis: With row-split restored (T3-009), the copy-engine AR (T3-004 code) on the side stream should hide more
              AR than NCCL (T3-002 microbench).
  change:     (worker leader-3.w1, ws t3-ce9) T3-009 + T3-004's ce_ar.cu / ce_all_reduce.py / cuda_communicator block /
              rowsplit._ce_comm. A = VLLM_CE_AR_MAX_BYTES=0, B default.
  results:    Observed: ABBA job 130: screen negative after 1 pair, B 886.8 vs A 909.0 (-2.45%). nsys job 132 (64 prompts,
              mixed steps, GPU0): NCCL side-stream AR ~half overlapped (256-512 rows: 31 of 58 ms exposed; >512: 58 of
              121); with CE the AR kernels vanish but no-kernel idle grows +24 ms (256-512) and +29 ms (>512): exposed
              comm (exposed AR + idle) A 34.7/97.7 vs B 33.7/77.5 ms per step; profile run-to-run noise (decode steps
              also 3% faster in B) prevents a conclusion.
  conclusion: Interpretation: the CE copies don't overlap with compute in situ: the stream waits on them (idle) instead of
              on NCCL kernels. Possible causes (untested): ~5 ops per AR vs 1 (more launches in eager >512 steps), copies
              captured in graphs, copy-engine queueing. Two server tests negative (T3-004 -1.56% all B<A, this one):
              dropped.
validity:     exact (bitwise = hostmem AR, T3-004 test); not proposed.
