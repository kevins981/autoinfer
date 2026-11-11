team:         T2
builds_on:    T2-002
performance:  "probe only: ablation of the lane interference at f=48"
summary:
  hypothesis: The +16.4 ms/step decode slowdown of T2-002 comes from a removable source (PCIe copies, cross-rank sync amplification, Marlin grid).
  change:     T2-002 + lane modes (VLLM_T2_LANE_MODE gemm/copy, VLLM_T2_LANE_ONLY_DEV) in t2_shadow_lane.py.
  results:    Observed (one job, 4 starts, 100 prompts c=16; agents/leader-2.w1/abl; OFF/full references cross-job from T2-002: ITL 25.3 / 41.8):
    gemm-only ITL med 42.9 ms (lane 42.8 TF); copy-only (paced ~80% of full duty) 29.7 ms; full lane on rank 0 only 39.4 ms; full lane with decode Marlin sms=128 43.3 ms.
  conclusion: Interpretation: the slowdown comes from the lane's GEMM thread, not its PCIe copies (+4 ms), not cross-rank amplification (-2.4 ms when only one rank runs it), not the Marlin grid choice. Leading suspect: CPU/GIL contention between the lane's Python thread (~300 launches/pass, per-layer syncs) and the worker's main thread; tested next with a CUDA-graph lane (T2-004).
validity:     probe only.
