# T3 map update 4 (resync 3, after sparring T3-1)

update-3 recommended closing. The sparring session (sparring/, T3-1) showed that two closing beliefs were assumed, not
measured:
- "k=3 is inherently too costly": T2-013 measured the step at +32%, but nobody had decomposed it.
- "Mixed-step gains are capped at 0.2-0.5%": that described the specific ideas left, not the ~7% of wall time that is
  still exposed AR.

## Tested since update-3
| test | record | result | evidence |
|---|---|---|---|
| k=2 vs k=3 profile | T3-015 | the +29% k=3 step is mostly implementation: GEMM M-tiling at 56/64 rows (+4.8 ms), DeltaLog 2-row cap (+2.6 ms); inherent AR +1.6, draft +1.0; no graph fallouts | job 142 |
| N-chunk row-split microbench | T3-014 | 3-4 chunks lose to 2. The per-layer mixer barrier costs 18-25%/layer. At 256 rows splitting loses (+41%), at 512 it wins (-6.5%) | job 141 |
| row-split threshold 448 | T3-016 | mixed, -1.45% (wide spread): inconclusive, leaning negative | job 143 |

## Framework lessons
- **Speculate knob: price the implementation's row-count discontinuities, not just rows.** In my 4-knobs pricing, an
  extra verify row cost link time only (the GEMMs are weight-bound). That holds only while the GEMM tiling covers the
  rows in the same number of M-tiles, and while every state kernel supports the row count (the DeltaLog cap). At 48 -> 56
  rows, both thresholds were crossed. Napkin per row is fine, but the cost is a step function.
- A team-level "dead" verdict (k=3: T1 twice, T2 once) can rest on unexamined implementation costs. A profile is cheaper
  than a fourth A/B.

## Current ranking
1. k=3 with M=56/64 GEMM configs (worker task 9: set up, not run), then DeltaLog with 3 logged rows. Napkin: decode steps +10-13%,
   overall maybe +5-8%, if TTFT p99 stays under the gate.
2. Row-split threshold: 128 lost (T3-010), 448 inconclusive (T3-016). Leave it at 256.
3. Request-level micro-batching to remove the mixer barrier in mixed steps: up to ~3%, large build (DBO excludes hybrid
   models). Not planned with the current budget.

## Closing (run ended by the human)
Unfinished when the run ended: k=3 part 1, the M=56/64 GEMM configs. The workspace t3-k3 is built and the sweep script
is ready (agents/leader-3.w1/k3_gemm.py, task9_status.md), but the job was cancelled before it ran. Part 2, DeltaLog with
3 logged rows, was not started. **This is the best open lead in the run for a successor.** Expected decode-step gain is
+10-13%, maybe +5-8% overall. The riskiest assumption is the TTFT p99 gate: k=3 broke it at 1989 ms in T2-013, and the
cause is not pinned down.
