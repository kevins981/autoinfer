team:         T1
builds_on:    T3-004
performance:  "unofficial: mixed, -0.31% vs T3-004 (within noise)"
summary:
  hypothesis: Our W8A16 Triton kernel runs at 0.77-0.90 TB/s at decode M (job 187-190). Tuned per-shape configs plus a BK=256
              path and deterministic split-K (for down at M=64) save time at M<=64. Sweep: -0.37 ms/step at M=48; op-level check
              (job 192): -0.11 ms/step at M=48 (~0.3% of wall).
  change:     w8a16_triton.py only (on T3-004): 7-tuple config table (BM,BN,BK,stages,warps,split_k,evict_first) with the
              sweep winners at M<=64 (a winner was kept only if >=1% faster). Kernel gains BK=256 (two 128-K sub-steps, one
              scale each), optional split-K via grid axis 2 with a fixed-order fp32 reduce kernel (no atomics), evict_first
              option, tl.multiple_of hints. M stays do_not_specialize; warm-up covers every distinct config including the
              reduce. Worker t1-w1, workspace t1-w16 (t1-w15 on T1-019 for the measurements).
  results:    Observed (job 192):
              - per-step GEMM sum old -> new: M=16 14.29 -> 14.24, M=32 14.35 -> 14.19, M=48 15.08 -> 14.97,
                M=64 17.57 -> 17.34 ms.
              - max |new-old| 0 except down 3.9e-3 (split-K reorder); deterministic across calls.
              - 0 post-warm-up compiles (check8, 5 shapes x 168 M).
              unofficial #1: B 862.3 / 869.9 vs HEAD mean 868.7 (mixed, -0.31%); p99 TTFT 944 / TPOT 37.5 ms.
  conclusion: Interpretation: the ~0.1 ms/step (0.3%) op-level gain is below the eval's resolution; not re-run. Parked: the
              W8A16 kernel family is at 0.8-0.9 TB/s and config tuning has no measurable headroom left.
validity:     W8A16 = same math (bf16 activations, fp32 accumulate); only fp reordering. No activation-format change, so no FP8
              A/B needed. Deterministic. Reviewed by leader-1.

unofficial #1 (2026-11-15T13:12): mixed: -0.31% vs HEAD T3-004 (B 862.3 / 869.9 vs reference mean 868.7 over 2 starts, 0 run in this job; 200 prompts); p99 TTFT 944 ms / TPOT 37.5 ms, MTP acceptance 2.341 (unofficial/1/unofficial.json)
