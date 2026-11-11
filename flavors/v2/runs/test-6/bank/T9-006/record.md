team: T9
builds_on: T1-042 (microbench only, no served change)
performance: "microbench: decode copy-engine AR is SLOWER than champion hostar+prefetch by 23-30 us/AR; gate (D_pf >= 8 us/AR better) FAILED"
summary:
  hypothesis: DMA AR keeps ~x1.0-1.1 under the L2-prefetch HBM traffic where SM hostar slows x1.45 (T8-001), so DMA + separate prefetch kernel beats champion hostar at decode (M=48).
  change: bench5.py / pfk.cu / job.sh (this folder); 2 procs, NUMA node 0 (shm pages all node 0, proof in out_full_r*.log), 16 layers = 32 ARs, real FP8 Marlin, CUDA graphs, 6 interleaved reps. Jobs 107 (smoke), 108 (full), ~0.5 GPU-min.
  results: Observed (ms/16-layer chain r0 | r1; per-AR us = (A_pf - arm)/32, negative = arm slower):
    N (no AR) 4.240 | 4.105; A_nopf 5.408 (A_pf - A_nopf = -15.0 / -15.2 us, prefetch is worth 15 us/AR);
    D_pf2 5.666 | 5.654 (-23.1 / -23.0 us); D_pf4 5.894 | 5.883 (-30.2 / -30.2); D2 6.120 | 6.128 (-37.3 / -37.8); D4 6.310 | 6.310 (-43.2 / -43.5).
    A_pf - N = +21.4 / +25.4 us per AR (ceiling).
  conclusion: Interpretation: the DMA path is uncompressed and pays several copy-engine round trips; its per-op latency (D2 ~37 us slower than A_pf) outweighs any HBM-contention advantage. A separate prefetch kernel recovers ~14 us for DMA (D2 -> D_pf2), the same as for hostar, but D_pf2 is still 23 us/AR behind the champion. More pieces is worse (4 pieces 7 us/AR worse than 2). Gate FAILED by ~31 us/AR; direction dead for decode M=48 unless DMA latency fell by >25 us (compressed payload or a single fused D2H/H2D). Not timeline-profiled, since far from the gate.
validity: measurement only.
