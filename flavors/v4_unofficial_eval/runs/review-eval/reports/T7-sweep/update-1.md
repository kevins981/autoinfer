# T7 map update 1 (leader-7, ~07:10 UTC)
## Tested
| candidate | result | evidence |
|---|---|---|
| #1 k=3 drafts + KV 5.5 GiB on the champion | 888.9 tok/s, stopped after 1 start (-0.7% vs 895.4): screen negative / ~neutral. Acceptance 2.70 vs 2.35 (+15%), median ITL 32.8 vs 28.7 ms (+14%) | bank/T7-001, GPU job 123 |
## Verdicts changed
- k=3: my napkin priced the extra step cost at +2.6-3.1 ms (+10%); measured ~+4 ms (+14%). The FP8 draft head and DeltaLog since
  T1-009 did not change the balance. Not "memory-blocked" (16 running at 5.5 GiB, no OOM at startup), but not a win either.
  It needs a cheaper per-row step (the 64-row verify) as well as freed spec-state slots.
- Calibration lesson: per-row costs at M=48->64 (AR, attention, GDN, sampling, KV at 98.6%) were underpriced; the napkin's
  "~75% of the step is fixed" holds for weights, but the row-proportional part is larger than I assumed (~4 ms per 16 rows).
## Current ranking
1. Free spec-state slots under DeltaLog (structural, >1 h). 2. DeltaLog for mixed-step spec rows (T5 #1, +0.5-1%, >1 h).
Neither fits the time left before 08:15 with a review. Review tool: backend error twice, map not reviewed.
