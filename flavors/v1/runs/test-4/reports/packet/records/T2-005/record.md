team:         T2
performance:  "unofficial: flat vs champion (580 vs 576 tok/s under nsys, 48 prompts); TTFT p99 2185 ms fails the gate"
summary:
  hypothesis: on the champion (T1-003, host AR + cheap drafter), depth 3 now pays off; profile why its step costs more than estimated.
  change:     T1-003 code (diff = T1-003's), num_speculative_tokens=3. Builds on T1-003. Profiling job (leader-2.w2 job 19).
  results:    Observed (nsys node-level, P2 vs prof_t1003, ms per steady decode step): Marlin 18.46->20.60, AR 5.88->6.90, GDN
              3.84->4.63, other 1.59->3.53 (3rd draft pass + eager glue); step period 36.95->41.59 (+12.6%, not +21% as in
              round 1); tokens/step 2.40->2.80 (+16.7%) => ~+3.6% on decode steps only. Prefill-containing steps (35-38% of wall,
              2048-token chunks) unchanged, so end-to-end flat. TTFT median 334->938, p99 1534->2185; mean TPOT 25.4->21.3.
              Also: the ~1.2 ms GPU gap before each target graph seen in node-level nsys is a tracing artifact (graph-level:
              cudaGraphLaunch 121-127 us). Real inter-graph idle in the champion: 1.57 ms/step (4.5%), ~43 eager kernels/step.
              Report: agents/leader-2.w2/REPORT_job1.md.
  conclusion: Interpretation: depth 3 is not a lever on this system: the extra verify cost eats most of the extra accepted
              tokens and prefill-heavy steps dilute it; TTFT degrades (cause not established; queueing behind 2048-token
              prefill chunks suspected). Closed.
validity:     leader-2: flag-only change on a reviewed diff; exact speculation. Not pursued.
