team:         T2
builds_on:    T3-009
performance:  "screen negative (1 pair, -11.71%), job 133"
summary:
  hypothesis: (Prompted by spar T2-1.) T1's k=3 losses (T1-003 -3.6%, T1-010 -5.8%) predate the W8A8 GEMMs, DeltaLog,
              the reduced-vocab draft head and the faster AR. Per-position acceptance at k=2 is 0.77 / 0.545, so a
              3rd draft might lift acceptance to ~2.67-2.7; break-even is ~2.55 at an estimated +10% step cost.
  change:     serve args only: --speculative-config num_speculative_tokens 3. Worker audit (GPU-free): DeltaLog caps at
              2 logged rows (kMaxLogRows=2), so at k=3 it falls back safely to full-state writes (DeltaLog's gain is
              lost). The other paths are row-count generic.
  results:    Observed: ABBA job 133 on ws t2-k3 (= T3-009): A k=2 928.9 tok/s, acc 2.323, TTFT/TPOT p99 929/27.6;
              B k=3 820.1 tok/s, acc 2.698, TTFT/TPOT p99 1989/29.9. Stopped after 1 pair.
  conclusion: Interpretation: acceptance matched the napkin (+16%), but the step cost rose ~32% (2.698/2.323 / 0.883).
              Even giving back DeltaLog's ~4% (an extension to 3 log rows) leaves k=3 ~-7%. TTFT p99 1989 ms also breaks
              the 1.2 x 1341 = 1609 ms gate. k=2 stays optimal for this drafter on current code. Don't extend DeltaLog to 3 rows.
validity:     n/a (negative; spec decode exact).
