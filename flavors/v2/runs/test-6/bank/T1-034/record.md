team:         T1
builds_on:    T1-030
performance:  "unofficial: probe only, no serving change; predicted net ~0 (-0.3..+0.5 ms/step) -> killed"
summary:
  hypothesis: Marlin FP8 GEMMs at M=48 have spare ALU so a PFOR-coded (lossless, ratio ~0.905) FP8 weight stream would cut HBM bytes and decode time (R12-B idea 2).
  change:     none to the engine. Standalone FP8xbf16 Marlin from T1-030 csrc (incl. {64,128,128} decode override) with a PROBE_K hook adding k masked integer ops per B word (SASS-confirmed +256/+512/+1024 instr/loop for k=4/8/16; outputs bitwise equal to k=0). Probe code: agents/leader-2.w22/probe/ (log probe/r1/log.txt), job 137, 0.2 GPU-min.
  results:    Observed (us/GEMM, median of 3, spread <= +-1.2 us; % vs k=0 for k=4/8/16):
              gate_up M=48 cold 107.6: +0.1/+0.5/+0.9 ; warm (88 MB prefetch) 84.6: +2.7/+4.3/+14.0 ; M=320 434.8: -0.4/+2.7/+3.3
              down M=48 cold 54.7: +0.7/+1.6/+3.0 ; warm 45.3: +3.4/+8.1/+19.0 ; M=320 240.7: -0.8/+2.0/+2.3
  conclusion: Interpretation: cold M=48 has ALU room, L2-warm does not (issue/tensor bound, consistent with KB28). Realistic PFOR decode ~8-10 ops/word -> k=8 row or worse. Net/step: down_proj (unprefetched, ~3.5 ms) ~-0.26 ms; prefetched GEMMs (~10.8 ms) +0.5..+0.8 if warm, ~-0.8 if cold; overall ~0, plus +2-3% Marlin on prefill. Killed. Down-only variant ~-0.2 ms/step minus prefill cost: not worth a codec.
validity:     N/A (no engine change; probe only).
