# T37 write-up (moonshot, leader-37)
Direction: one static row-budgeted step ("fill every weight pass"). Result: champion T37-002 (1142.2 official; same-job ABBA +1.34% vs T31-003).
Records: bank/T37-001 (measurements, step logger), bank/T37-002 (champion). Notes: local_kb/leader-37.md. Proposal: proposals/T37-1.md.

## Supported by measurement
- Decode step: T = 16.9 ms + 0.10 ms*rows + 0.83 ms*draft steps up to 64 rows (jobs 368/369/372).
- k=3 with 16 running (two-slot GDN state, fixed): +1.34% same-job (job 375), not the +4.5% the cost model predicted.
- Host-side launch removal in mixed steps does not pay (T32-003).
- Confidence-budgeted depth lost in the implementations tried (T5-006, T27-001/003).
- Page-boundary defect in the two-slot code (KB110); lesson: when touching GDN state, test requests longer than 1600 tokens.

## Assumed, NOT established (leader's scope for the stop, after sparring T37-1 turns 3-4)
- That the all-reduce volume (10 KB per row per all-reduce, 128 per step) is a floor: argued only for the champion's tensor partition; no other partition or collective schedule derived or prototyped. First thing to test: derive the minimum cross-GPU information from the computation's dependencies, independent of today's split.
- k=4 at ~+1% (needs replay log cap kMaxLogTokens lifted in the kernel).
- A fully static mixed step at 2-4% (never measured as equal-work mixed vs separate); the ~15 ms per prefill event attributed to GPU work is inferred from kernel sums plus T32-003, not reconciled on one timeline.
So: T37-1's tested directions are closed with evidence; a broader "no structural gain above 5% remains" is outside what was shown.
