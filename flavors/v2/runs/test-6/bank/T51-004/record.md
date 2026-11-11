team:         T51
builds_on:    T51-003 (workspace leader-51.pig, + pad-free bs_gemm configs 31/34/35/134/135 BM=128/160/192)
performance:  "microbench (job 439, 1.1 GPU-min): pad-free big tiles lose (gate_up M=160 190 vs 157.7 us); ncu refutes an L2->SM bound"
summary:
  hypothesis: the TP-shape GEMM band at M=128-192 is L2->SM-traffic-bound (64x64 tiles re-read A per n-tile), so
              pad-free one-m-tile / stream-K kernels would reach ~0.055 ms/row.
  change:     ncu (--cache-control none) on gate_up cfg 127/103/31/34/35 at M=128/160/192; standalone sweep of the new
              pad-free configs. Files agents/leader-51.w1/m1/{sweep_padfree.json, ncu_*.csv, ncu_gu.ncu-rep, kernel_read.md}.
  results:    Observed: cfg127 L2 throughput 55-70% of peak, SM 26-35%, issue active 26-29%; top stalls: long_scoreboard
              at M=128, math_pipe_throttle at M>=160. Big-tile configs (1 CTA/SM) 33-42% L2, slower: gate_up M=160 cfg34 190 vs
              cfg127 157.7 us; M=192 cfg135 198 vs cfg103 159.9; M=128 cfg31 135 vs 122.6. down/gdn_out pad-free worse;
              qkvz cfg134 68.6 vs 73.7 at M=160 (only win, ~0.3 ms/step).
  conclusion: Interpretation: no single resource is saturated; at M>=160 the kernel is tensor-issue limited at ~74% of the
              e4m3 roof on padded rows, at M=128 load-latency limited. Scheduling fixes (stream-K on the cfg127 tile) bound to
              the tail: est. band 0.06-0.075 ms/row (unbuilt). The 0.055 gate is out of reach without a kernel above ~85% of
              roof, which no full kernel in this run has reached (KB83/KB98/KB101). Stream-K kernel not built.
validity:     microbench only.
