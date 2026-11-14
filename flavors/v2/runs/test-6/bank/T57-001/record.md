team:         T57
builds_on:    T52-005 (measurement only, no code change)
performance:  "measurement: no speedup claimed"
summary:
  hypothesis: the champion's mixed steps (~51 ms for <256 rows vs ~24 ms decode, T51/KB131) are host-enqueue bound, so capturing more of the mixed step would pay (sweep candidate #2, reports/T57-sweep.md).
  change:     none; T29_HOSTPROBE on the champion code (workspace leader-57.champ), 200 prompts c=16, 1 trial, rank 0.
  results:    Observed (job 484, 2.2 GPU-min; agents/leader-57.w1/task1/REPORT.md): 1334 steps; GPU idle between steps 0.01 s of 41.4 s (p1..p99 0.00 ms); host runs ~1 step ahead.
              GPU span per class (n, mean, median; median host enqueue): decode n=1163 23.8/24.6 ms, host 1.5; mixed <256 rows n=95 51.0/46.4, host 21.5; 256-767 n=59 105/99, host 36.7; >=768 (waves) n=17 160/147, host 125.5.
              In-step GPU-waits-for-host bounded above by 0.38 s (<256) + 1.12 s (256-767) per 41.4 s (loose bound). Probe layer events don't fire with T32's captured core on.
  conclusion: Interpretation: T51's "51 ms" small mixed step is GPU time, not host feeding. Capturing more of the mixed step can only pay by shortening GPU work (launch gaps inside the step), ceiling <=~3.6% e2e by the loose bound, probably far less. Candidate #2 demoted. Wave steps (host 125 vs GPU 141 ms) remain near host-bound but are a thin sample (n=17).
validity:     measurement only.
