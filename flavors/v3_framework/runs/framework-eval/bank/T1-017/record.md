team:         T1
builds_on:    T1-014 (code of workspace t1-b; no change)
performance:  "diagnostic (no serving change)"
summary:
  hypothesis: Marlin decode GEMMs run at 0.66 of BW in situ (nsys, 141 us gate_up) vs 0.77-0.82 isolated (ncu, 115 us).
              Spar T1-1: the timing methods differ, so the gap may not be real. If real, is it L2 eviction by neighbours / clocks?
  change:     none. Microbench agents/leader-1.w1/marlin/bench.py (real prepare_fp8_layer_for_marlin/apply_fp8_marlin_linear,
              M=48, random fp8 weights, 128-block scales, 1 GPU), worker job 68 (0.2 GPU-min).
  results:    Observed (CUDA events; SM 2655 MHz, mem 10251 MHz): gate_up 89 MB: 115.7 us same weights, 114.7 us 12 cold layers
              cycled (777 GB/s); down 44.6 MB: 39.9 us same weights (L2-resident, 72 MB L2), 58.4 us cold (763 GB/s).
              CUDA graph of 12x(gate_up, down): 173 us/layer without, 169 us/layer with a 100 MB memcpy per layer
              (gemm-only = graph minus copy-only graph). Eager with per-kernel events + memcpy: 148.5 / 73.7 us.
  conclusion: Interpretation: ncu's isolated 115 us is real and holds for cold weights and inside a graph next to 100 MB of
              other traffic. So L2 eviction, clocks or Marlin itself do not explain the in-situ 141 us. Remaining suspects:
              the second rank / host-memory AR spinning blocks and PCIe traffic, power under full 2-GPU load, or nsys's
              timing inside the real graph. Next cheap test: the same microbench on both GPUs with a concurrent AR loop, and
              nvidia-smi power/clock during a serving run.
validity:     no serving change.
