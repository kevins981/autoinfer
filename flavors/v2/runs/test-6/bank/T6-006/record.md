team:         T6
builds_on:    T6-004
performance:  "INVALID (KB61): ws t6-pol.t was never built; its _C is the baseline prebuilt .so (symlink to .cache/.../prebuilt, 35,755,000 B) => no GDN replay, no Marlin tile/policy."
summary:
  hypothesis: decompose mixed-step cost (B default budget, C --max-num-batched-tokens 96) into host vs GPU with T3 timers + host stamps.
  change:     ws t6-pol.t = t6-pol + T3 timers (created --from-ws, NOT rebuilt).
  results:    Observed (job 56, 5.8 GPU-min): decode fwd 25.6 ms at M~45 (vs 22.35-22.40 in built runs: consistent with no GDN replay); C mixed M=96 fwd 55.3 / period 58.6 vs decode 24.1-25.6; GPU period ~= sum of stamps (no inter-step idle); PIECEWISE host time inside forward ~= GPU forward.
  conclusion: Interpretation: absolute numbers invalid (baseline kernels). Qualitative points that do not depend on the kernels: no idle between steps (agrees with T2-006); a PIECEWISE step's host enqueue time ~= its GPU time (launch-bound vs run-ahead not separable without a kernel trace).
validity:     INVALID for timing; kept for the record. Lesson: --from-ws copies of a built workspace also need `ws build` (KB61 applies).
