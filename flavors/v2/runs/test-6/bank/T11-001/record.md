team:         T11
builds_on:    T4-001 (T1-042 + MARLIN_SMS grid override; ws t11-lane, built)
performance:  "microbenchmark only: premise FAILED (window lane). Prefill GEMM slabs in decode AR windows slow the chain +17..33% for less work than that"
summary:
  hypothesis: During each decode TP all-reduce (~46 us, 8 AR + 8 L2-prefetch CTAs) ~110 SMs' tensor cores are idle; compute-bound prefill GEMM slabs (M=256) forked into those windows would run nearly for free, letting small prompts be prefilled in the background instead of in mixed steps (napkin payoff ~+10%).
  change:     2-process microbench by leader-11.w1 (jobs 94, 95, 96; 1.6 GPU-min), agents/leader-11.w1/lane.py (copy here). T4-001 chain (16 layers = 32 ARs, real FP8 Marlin per-rank shapes, CUDA graphs, 6 interleaved reps), NUMA-bound to node 0 (slots verified on N0 via numa_maps). Lane = FP8 Marlin M=256 K=5120 on a side stream, forked after each AR producer, joined before the consumer; cold weight pool >= 400 MB; grid S=112/96; N=512/256/1024 (~22-31 us standalone). Arms: A_pf (champion), A_nopf, L (lane + pf), U (lane unjoined), Lnp (lane, no pf).
  results:    Observed (rank 0, ms/chain; r1 agrees within 0.05):
    A_pf 4.917; A_nopf 5.423; no-AR 4.236.
    L (lane+pf): +0.99..+1.61 ms (+20..33%) vs A_pf; U (unjoined) +0.83..+1.38; Lnp +0.57..+1.19 vs A_pf (= +0.07..+0.68 vs A_nopf).
    Lane GEMM in-window 65-91 us vs 22-31 us standalone (x2.3-3); AR 45.7 -> 59-85 us beside the lane; consumer GEMMs unchanged (prefetch benefit survives ~97-100%); AR-end -> consumer gap 3 us (N=256) to 14-36 us (N=512/1024).
    Smallest lane kernel ~22 us (launch/latency floor at M=256).
  conclusion: Interpretation: the AR windows are not usable idle compute on this machine. The lane competes with the SM-driven host AR (x1.3-1.9 slower) and itself runs 2-3x slower, so decode loses more than the lane gains. Valued at the real prefill rate (~125 TF), the lane's FLOPs are worth 0.34 ms (N=512) / 0.69 ms (N=1024) per chain, against costs >= 0.57 ms even in the best arm (Lnp, which also gives up the 0.51 ms prefetch). Unjoined is only ~0.18 ms better than joined: interference, not drain. Extends KB59/KB62 (zero-sum) to temporal fill of the AR windows. Direction closed.
validity:     measurement only; no served change (change.diff = T4-001's MARLIN_SMS override, env-gated).
