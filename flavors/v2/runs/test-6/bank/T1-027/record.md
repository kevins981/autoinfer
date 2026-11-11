team:         T1
builds_on:    T1-026
performance:  "measurement: split 256 (=T1-026) 840.8/841.5 vs split off 825.2/823.7 (+2.0%) vs split 128 832.2/835.2 (-0.9% vs 256); not for eval"
summary:
  hypothesis: with the AR slots on the near NUMA node (T1-026) the R7-A token split may pay differently; threshold 128 may cover more
              mixed steps (w15 also showed the R7 "12 GB/s in-chain DMA copies" were a job-wide host-traffic slowdown, not DMA-specific:
              in-chain copies run 20-24 GB/s; node-0 vs node-1 slots: hostar alone M=300 229 vs 315 us).
  change:     none (T1-026 code); env-only arms: VLLM_DMA_SPLIT=0 / VLLM_DMA_SPLIT_MIN_TOKENS=128. Worker leader-2.w15, job 102,
              agents/leader-2.w15/abc/abc1, NOTES.md; microbench jobs 98-99 (copy/c1, chain/k1_*).
  results:    Observed: A1 B1 C1 C2 B2 A2 (200 prompts c=16, all bound to node 0): A 840.8/841.5, B 825.2/823.7, C 832.2/835.2;
              within-arm spread <= 0.4%. TTFT p50 A 187/190, B 188/197, C 207/193; TPOT p50 16.6-16.9 all; acceptance 2.33-2.35.
  conclusion: Interpretation: R7-A split +2.0% with near-node slots (vs +0.7% in R7's noisier ABBAs); threshold 128 worse (extra Marlin
              launch + join cost on 128-255-token steps). Keep 256.
validity:     measurement only.
