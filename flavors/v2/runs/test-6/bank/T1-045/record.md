team:         T1
builds_on:    T1-042
performance:  "unofficial: null: thresholds 192 / 128 are -0.5% / -0.6% vs 256 (each below A in both pairs); keep 256"
summary:
  hypothesis: T1-042 made the DMA lane ~20% cheaper; lowering VLLM_DMA_SPLIT_MIN_TOKENS from 256 to 192 or 128 might now pay on small mixed steps (exposed hostar ~19 ms per <192-token step in the anatomy).
  change:     none (env-only arms on T1-042 code, ws leader-2.baproj). Threshold verified from both workers' /proc/<pid>/environ in every arm.
  results:    Observed (job 183, fresh servers, 200 prompts, c=16, order A B C C B A):
              - tok/s: 256 897.4 / 895.3; 192 895.8 / 888.5; 128 888.7 / 893.8
              - means: A 896.4, B 892.2, C 891.3
              - ITL p50 25.32-25.39 in all arms
              - p99 TPOT 30.6-41.5 with no pattern
  conclusion: Interpretation: consistent with T1-027: the extra Marlin launch, 64-row remainder and stream join outweigh the AR time hidden on 128-255-token steps. Keep 256.
validity:     N/A (no code change).
