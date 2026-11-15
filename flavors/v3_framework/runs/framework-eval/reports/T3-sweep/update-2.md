# T3 map update 2

## Tested since update-1
| candidate | record | result | evidence |
|---|---|---|---|
| in_proj prefetch, repeat A/B on T3-003 | T3-005 | all B > all A, +1.86% | job 108 |
| in_proj prefetch ported onto T2-009 | T3-006 | **official 919.0 (+1.33% vs T2-009): champion** | ABBA job 110 (+1.86%, all B > all A) |
| DeltaLog with 2-step checkpoint | T3-007 | ceiling ~0.33 ms/step before replay costs: not built | job 109 |
| evict_first / .cs state stores | T3-007 | no gain (within +/-3 us/layer) | job 114 |
| mixed-step profile of T2-009 | (analysis) | every mixed-step NCCL AR serialized on the main stream: row-split was off | job 117 |
| row-split restored with fused norm+quant (on T3-006) | T3-008 | +1.58%, all B > all A | job 121 |
| same, ported onto T2-011 | T3-009 | **official 951.5 (+2.11% vs T2-011)**: champion candidate | ABBA job 123 (+1.40%) |

## What changed in the map
- **A regression dimension the framework doesn't name: the interactions between knobs across teams.** T2-009's Rewrite (a
  fused norm+quant) silently turned off a Window choice (row-split overlap) through a guard condition. The framework
  prices each knob site independently; in practice, a change at one site can switch off another site's choice. Cheap
  check adopted by T2 and T3: grep every server log for the "row-split active" line.
- My "rank skew dominates mixed steps" hypothesis (update-1) was wrong. In T2-009 code the skew is 30-55 us per AR
  (5-15%); the exposure came from no overlap at all. T3-004 (CE AR, -1.56%) ran on T2-008, where row-split was active.
  Its cause is still unexplained.
- The link floor of the SM-issued hostmem AR is lower than my copy-engine 22.6 GB/s duplex figure. T2 measured 31-34 GB/s
  total for concurrent SM-issued write+read (T2 job 122). T2 is working on a pipelined AR (~35 us target), which also
  shrinks my prefetch window: the two need joint tuning.

## Current ranking (napkin vs the T3-009 code)
1. Prefetch retune after T2's faster AR, if it lands (window 44 -> ~35 us; 40 MB may no longer fit): small, cheap.
2. Mixed steps < 256 rows have no row-split (31 of 64 mixed steps in job 117, AR 27% of their time). Lower
   ROWSPLIT_MIN_TOKENS (e.g. 128) now that it works: env-only test, ~0.5-1%.
3. Copy-engine AR for the side stream on top of the restored row-split (T3-002 says it overlaps better than NCCL;
   T3-004's loss is unexplained): a re-test on current code is cheap since the code exists. Up to ~1-2%.
4. The <256-row mixed-step idle (~2.5 ms/step real gaps per T2): ~0.9%; T2 dropped it.
