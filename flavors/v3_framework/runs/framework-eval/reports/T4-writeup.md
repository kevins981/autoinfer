# T4 write-up (sweep team, leader-4 + worker leader-4.w1)

**Result.** No official records and no bank records. The run was paused, then ended, before T4's first experiment ran.
GPU used: ~6 of 100 GPU-min, on two nsys profiles of the champion T3-009 (jobs 139 graph-trace, 140 node-trace).
Map: reports/T4-sweep/ 1-6 + update-1.md; summary reports/T4-sweep.md.

## What we learned
1. **Price mixed steps per prompt-length bucket, from the real prompts.** Mixed steps come in three regimes:
   - small P: CPU-bound in the eager GDN core;
   - 256-512 tokens: GPU-bound and graphed;
   - over 512 tokens: fully eager.

   I tokenized the benchmark's 400 prompts on CPU only, with the harness's sampler and seed (agents/leader-4/plen.py, plen.json):
   - total 104.5K prompt tokens, median 128;
   - 232 prompts < 211 tokens, 92 mid, 76 whose step exceeds 512 tokens.

   With per-bucket step costs from the profile, the napkin reproduces the official wall time: 96.9 vs 95.9 s.
2. **Where the time is, on T3-009.**
   - Decode-only steps: ~55 s (26.5 ms each).
   - Mixed steps: ~42 s, against a ~22-25 s floor of max(weights, rows x 24.3 GFLOP / 150 TF/s) + 5 ms.
   - Every decode candidate on my map was already built or measured at its floor by T1-T3: achievable DRAM ~870 GB/s,
     k=3 dead, hostmem/CE AR variants dead, ReplaySSM < 1%.
3. **Earlier verdicts carry the code they were priced on.**
   - ">512-row steps are GPU-bound, graphing gains nothing" (T2) used a Marlin-era per-row slope (0.25 ms/row).
   - On the W8A8 champion a graphed step costs ~0.16 ms/row, and the worker's graph check shows > 512-token steps are
     entirely eager (219 ms vs ~84 ms).
   - Re-pricing old "dead" verdicts on the current champion is cheap and is where my map found its top candidate.
4. **nsys can't settle CPU-bound questions.** My graph trace shows ~1.4 ms CPU vs ~0.2 ms GPU per eager GDN layer in
   small mixed steps; T2's gap analysis says only ~2.5 ms of real idle per step. Both come from nsys. The tie-breaker is a
   non-nsys per-step GPU-event log, which I prepared but couldn't run.

## Framework notes
- The framework's workload input is one snapshot. Here the objective depends on a *distribution* of snapshots (prompt
  lengths), and the regimes split by bucket. Adding "price over the snapshot distribution" exposed both of my top
  candidates.
- No CPU/launch resource in the window cost. I added "host launch time" as a fourth term in max(); T1/T2 hit the same gap.
- Use measured achievable peaks, not spec (T2's lesson). Without them I mis-read decode GEMMs as 15% inefficient.

## Hand-off (for a later run): see reports/T4-sweep/update-1.md
1. CUDA graphs 576..1024: +5-6% napkin. Script ready: agents/leader-4/e1_job.sh (~20 GPU-min).
2. Small mixed steps' eager GDN core: +9-11% if real. The same job's step logs decide it (agents/leader-4/buckets.py).
3. Mixed-step spec rows on the CUDA MTP kernel: ~+0.8%.
Workspaces: t4-champ (= T3-009), t4-steplog (T3-009 + T2's step logger, built).
