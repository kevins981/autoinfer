team:         T2
builds_on:    T2-012
performance:  microbench only: no gain (Humming FP8 W8A16 = Marlin within ±5% at M 16-48)
summary:
  hypothesis: KE1 cheap probe: the engine's HummingFP8ScaledMMLinearKernel (selectable with --kernel-config '{"linear_backend":"humming"}') streams FP8 weights faster than Marlin at decode M.
  change:     none in the engine (microbench, worker t2-w1, jobs 59-62; agents/t2-w1/m6/hum.py, outh.log).
  results:    Observed: real block-FP8 weights, L2 flushed, CUDA graph x20, 1 round. M=48 Marlin/Humming us: qkvz 50.2/52.6, attn_qkv 46.7/45.7, out 22.1/22.6, gate_up 106.4/106.9, down 56.4/55.4; M=16/32 Humming 0-2.5% faster. Max |diff| 0.03-0.06. Both at 0.80-0.88 TB/s. Humming JIT (NVRTC) needs libnvrtc-builtins on LD_LIBRARY_PATH and writes ~/.humming (HOME override needed).
  conclusion: Interpretation: same bandwidth plateau as Marlin; no reason to switch (plus deployment hazards: $HOME cache, extra lib path). A faster decode GEMM would need a new kernel (>=2 blocks/8 warps per SM, lighter dequant, split-K); T1 measures decode GEMM at 1.07-1.14x the HBM floor (T1 job 48), so the ceiling is ~7-10% of GEMM time (~1 ms/step), high effort.
validity:     n/a (no change). Note: the worker's first job accidentally created ~/.humming (72 KB), deleted right after; later jobs set HOME inside the agent folder.
