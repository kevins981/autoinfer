team:         T42
builds_on:    T42-003 (no code change: profile only)
performance:  "unofficial, nsys (job 414): decode step period 24.94 vs 25.14 ms (-0.20 ms, -0.8%); lm_head+sample critical
              path 796 vs 1065 us median (-270 us). Below the ~0.3 ms/step stop line; stopped after the profile as instructed."
              (DRAFT by leader-42.w1)
summary:
  hypothesis: T42-003's mixed ABBA (+0.53%) comes from CPU-side overhead of the new eager sample path (extra launches,
              the LB all_gather). If so, fixing it (graph capture or fusion) would recover the ~0.42 ms that stage 1
              predicted.
  change:     None. One nsys job: profile_nsys.sh, 48 ShareGPT prompts at c=16, A = T37-002 (workspace t42-a) and
              B = T42-003 (t42-cs) in the same job, TRACE=node. Analysis scripts in profile/:
              - sectstat.py: per decode step, from the first lm_head kernel (A: lmh_gemm_kernel, B: hprep_kernel) to
                the start of _greedy_ids_rejection_kernel on GPU 0, plus median time per kernel name;
              - sect.py: one step's kernel sequence with gaps and launch->start lag.
  results:    Observed (GPU 0 = rank 0; medians over ~320 steps; nsys adds per-launch overhead):
              lm_head + sample section, start -> rejection kernel:
                A 1065.3 us (p10 1042, p90 1086), kernel-busy 1055.0
                B  795.5 us (p10  663, p90  832), kernel-busy  779.9
              Per kernel, B:
                screen 647.8 | gather (host->HBM) 76.4 | rescore 13.5 | LB+pair all_gather (NCCL, 2 calls) 10.5 total
                hprep 6.7 | finalize 2.1 | fallback no-op (rawgemm early exit) 2.0 | fblist 1.8 | unmap 1.2
                keys_decode 1.0 | combine elementwise/cat ~12
              Per kernel, A: lmh_gemm 1025.4 | argmax 6.8 | all_gather 5.7 | combine elementwise/cat ~11.
              Rank 1 (B): screen 641.7, gather 1.2. The cross-rank LB leaves rank 1 almost no candidates, so rank 0's
                gather sets the critical path.
              Launches are not on the critical path in either arm:
                - CPU launch -> GPU start lag is ~18 ms (A) and ~51 ms (B): the CPU runs far ahead.
                - Inter-kernel gaps inside B's section add ~16 us (A ~10 us).
                - Decode-step GPU idle in window: A 0.20 ms, B 0.21 ms.
              Decode step period (step_anatomy): A 25.14 ms, B 24.94 ms.
              Stage-1 comparison (standalone, cold L2, M=64): champion 1178 vs R-HOST full path 759 us (-419).
                In situ: A's lmh_gemm runs 1025 us (153 us faster than standalone cold).
                B's screen runs 648 us (= standalone cold): the plane prefetch shows no measurable effect.
                Gather is 76 us (stage 1: 80).
  conclusion: Interpretation:
              - The hypothesis is wrong. The CPU-side overhead is ~6 us of extra gaps and no launch-bound time, so graph
                capture or fusing launches can recover at most ~10-20 us/step.
              - Saving vs stage 1: in situ it is 270 us/step, not 419. Most of the difference is that the champion's
                kernel gets faster in situ (-153 us, likely the final-norm AR's L2 prefetch of the encoded prefix and/or
                warm L2) while the screen does not.
              - Remaining levers and rough ceilings:
                (a) make the plane prefetch land (hint mode lands ~40% of 88 MB per KB97: <= ~35 us);
                (b) an HBM cache of exact rows in the freed 0.34 GiB to cut the 76 us gather. It needs a
                    non-benchmark-specific (dynamic) policy; the hit rate is unknown;
                (c) fusion: ~10-20 us.
                Best case ~0.33-0.4 ms/step, all of it uncertain except (c).
              - Per the leader's rule (stop if the saving can't clearly exceed ~0.3 ms/step), I stopped after the
                profile.
              - The measured -0.20 to -0.27 ms/step (~0.8-1.0%) is consistent with T42-003's ITL p50 (-0.23 ms) and its
                mixed throughput ABBA.
validity:     no code change (see T42-003).

leader review (leader-42, 2026-11-11): accepted as a measurement (no code change). Interpretation: the certified
lm_head saves 0.27 ms of the lm_head section (1065 -> 796 us) and 0.20 ms of the decode period (~0.8% decode, ~0.5-0.8%
e2e; consistent with T42-003's mixed ABBA +0.53%). The stage-1 -> in-situ shrink is the champion's in-situ advantage
(AR-window prefetch of its encoded prefix: 1178 cold -> 1025 in situ) that the plane does not get (hint-mode
prefetch lands ~40%). Not pursued further under the stop rule; open levers listed above (real evict_last loads from the
AR's extra blocks as in KB97 rather than hints; an HBM exact-row cache; a 6-bit plane with the global LB).
