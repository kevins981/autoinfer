team:         T2
builds_on:    T2-003
performance:  "probe only: CPU-free (CUDA-graph) lane, f=48: decode ITL unchanged at 41.9 ms (OFF 25.3, same job)"
summary:
  hypothesis: T2-003's slowdown is CPU/GIL contention from the lane's Python thread.
  change:     lane pass (64 layers GEMMs + copies) captured as one CUDA graph on the green-ctx stream; thread loops replay + stream.synchronize (VLLM_T2_LANE_GRAPH=1); optional sys.setswitchinterval (VLLM_T2_SWITCH).
  results:    Observed (one job, 3 starts; agents/leader-2.w1/g4): OFF ITL 25.3 / TPOT 16.55 / TTFT p99 1213; graph lane ITL 41.9 / TPOT 32.8 / TTFT p99 1904, lane 30.4 TF; + switchinterval 0.5 ms ITL 41.9 / TPOT 32.6 / TTFT p99 2410.
  conclusion: Interpretation: not CPU. Across T2-002..004, decode ITL vs lane SMs: f=32 36.8, f=48 41.8, f=64 46.8 ms, close to 25.3 x 128/(128-f) (33.7 / 40.5 / 50.6): the in-server decode step behaves as if it needs every SM in proportion. The pure-Marlin microbench (T2-001) had slack; the full decode step (AR, GDN, small kernels, Marlin) does not. Prefill is also SM-proportional (~1 TF/SM). So spatial SM sharing is zero-sum here: at f=48, decode runs at 0.61 of full speed and the lane at ~0.29 of the full-GPU prefill rate, sum 0.90 < 1 (a loss vs serialized). DIRECTION CLOSED (T2-1).
validity:     probe only.
