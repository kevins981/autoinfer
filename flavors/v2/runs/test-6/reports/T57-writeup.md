# T57 write-up (sweep, leader-57). Recommendation: CLOSE
Records: bank/T57-001..006. Sweep: reports/T57-sweep.md. Worker: leader-57.w1 (agents/leader-57.w1/task1..7/REPORT.md).
Sparring: T57-1 (turns 1-8). GPU: ~62 of 100 used (jobs 484, 486, 488, 492, 493, 496, 497, official T57-004, 501, 507, 508, 510, 511, 513).

## Result
- **T57-004 CHAMPION 1225.5 tok/s** (+1.08% over T52-005; ABBA job 497 +1.68%, all B > all A). Exact drafter package: T56's
  precomputed MTP fc-embedding table in pinned host memory (removes the drafter's embedding and its AR, halves fc), pack + one
  cross-rank argmax at the screened draft head, 1-tile mamba glue kernels. Later built on by T58-001.

## What the sweep found (framework applied; calibrated with T51's closed-loop sim: decode 57.4 s + prefill events 17.8 s/trial)
- #1 spatial prefill lane (framework Mechanism 2: compute-bound prefill beside DRAM-bound decode): fails. T57-002 microbench:
  f=48 gives lane 85 TF with decode at 1.40x, and an ideal DRAM stream starves the lane (DRAM/L2 arbitration, not SM count). T56-001
  in-server: 0.137-0.156 ms per lane token vs 0.167 joined. Closed.
- #2 mixed-step host overhead: T57-001: mixed steps are GPU-bound (GPU idle between steps ~0; host wait inside steps <=1.5 s/trial,
  a loose bound). T51's "51 ms small mixed step" is GPU time.
- Drafter: T57-003 anatomy (1.90 ms/step, ~0.76 ms above the byte floor) -> T57-004 (win). Second round T57-006 (radix top-64
  select bitwise, one-launch GDN flush bitwise on target state (job 513, 16 cases + negative control), host target embedding
  T57-005 bitwise, frees 1.17 GiB/GPU): -0.1 ms/step on the probe, null in a 6-start ABBA (+0.01%, job 510). Not submitted; the
  exact pieces can stack with a future memory-blocked change.
- Priced at a loss or too small, not built: k=4 (+10% tokens for ~+19% step at measured row costs); wave-step capture to lower
  T30_WAVE_MIN to ~384 (+0.2-0.5% realistic, days, host-bound risk); token-split threshold below 256; drafter on one GPU.

## Scope of the stop (sparring T57-1)
"T57 found nothing more it could resolve within its budget", not "nothing above 0.5% remains". Open and NOT bounded:
- GDN replay: ~2.0 ms/step vs ~1.2 ms fp32 state-read floor (~0.8 ms, ~2.5% e2e if fully closed). The evidence is failed attempts
  (L2 prefetch in the AR window, scheduling), not a floor. Cheapest test: per-layer in-situ replay latency vs bytes timeline.
- Decode all-reduce 6.7 ms: TP fixes the bytes (~72 MB/direction/step, ~3 ms per direction at ~24 GB/s); the write-then-read
  sequencing, startup and sync (~0.7 ms) are not bounded, only covered by five teams' duplex attempts.
- Whether ~0.5% gains are resolvable: estimate paired-difference variance before declaring them unmeasurable.

## Framework notes (for its authors)
The window cost model lacked: host launch cost as a resource, SM partitioning (overlap of F and BW is bought with SMs, and here
DRAM/L2 arbitration makes it zero-sum), per-transfer latency and half-duplex sequencing of host-staged links, the exact-accumulation
ridge (~80/170 rows, not 330) with m-tile staircases, and closed-loop dynamics (needed a simulator). Details: reports/T57-sweep.md §5.
