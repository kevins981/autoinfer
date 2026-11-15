# T1 map update 4 (HEAD = T1-015; CONFIRMED = T1-012 at 785.3 official)

## Tested since update-3
| candidate | record / job | result | verdict |
|---|---|---|---|
| Chunked GEMM+AR overlap, revisited on T1-010 | T1-012 | PASS +0.52%; confirmed in chain (785.3 official) | **won** |
| Cap max_num_batched_tokens 512 (T2's W1b) | T1-011 | -1.05% (2 starts) | dropped |
| Target lm_head via cuBLAS addmm path (s16816 instead of split-K s1688; microbench -150 us/call, job 99) | T1-013 (cancelled: T2-015 reverted), T1-014 | -1.29% (2 starts) | dropped; microbench gain did not carry over (cause untested) |
| Split-KV under the no-sync UB plan (T1-010 had disabled it; found by T2's profile, job 101) | T1-015; jobs 112, 113 | bitwise == exact split plan; ABBA +2.2%; PASS +1.29% | **won (HEAD)** |
| Draft lm_head tile / Marlin | job 99 (baseline only) | <=0.08 ms/step | not pursued |

## Map lessons
- When a candidate trades two costs (T1-010: removed ~2.4 ms of idle, added ~1 ms of attention), record the paid cost as a
  new candidate right away. I knew about the split-KV cost but didn't put it on the ranking; another team's profile found it.
- The microbench-to-real gap cuts both ways: T1-010's nsys gain shrank ~10x; T1-014's microbench gain turned into a loss.
  For changes worth <0.5%, the unofficial eval is the only reliable measure.

## Ranking now (against T1-015)
1. In-situ GEMM excess in GDN layers (~1.5-2 ms/step, cause unknown after jobs 89-91). Next test: a real 2-layer forward
   from the loaded model with and without the GDN layer in between. Hours of work; uncertain.
2. Re-profile T1-015 decode (graph-level + kernel) to re-baseline the budget. Cheap (~1-2 GPU-min); may show new paid costs.
3. Draft head tiles: <0.3%.

## Addendum: HEAD decode re-baseline (t1-w1 job 117, agents/t1-w1/budget/head15.md)
Profiled decode period 32.63 (T1-010) -> 31.66 ms (T1-015); target graph -0.82 ms (split-KV back); idle 0.12 ms/step.
Since T2-011 the decode graph content is flat (~29.3 ms); all gains came from idle (T1-010) and prefill (T1-012).
No component rose. Remaining gaps (from the T2-011 per-kernel budget): Marlin 17.6 vs 12.1 floor, drafter 1.7 vs ~0.45 weights-only (~0.3 recoverable),
GDN 3.7 vs 2.4, lm_head 1.58 vs 1.26.
