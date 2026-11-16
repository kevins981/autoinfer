# T7 write-up (leader-7, sweep team; no worker started)

## Result
- No win. T7-001 (k=3 MTP drafts + KV 5.5 GiB on T5-001's code, flags only): 888.9 tok/s, stopped after 1 start (-0.7% vs 895.4),
  screen negative. GPU job 123; bank/T7-001. GPU used 3.2 of 45 min.
- Map: reports/T7-sweep.md (files 1-6 + update-1). Compact: the team opened at 06:59 with the run ending 08:15.

## What I learned
1. **Capacity is a Speculate dimension here.** The GDN spec-state slots (k+1 fp32 states per request per GDN layer) count against
   the KV budget, so draft count trades against running requests. My phase-1 napkin priced k=3 in time only; the history (T1-004)
   caught it before the GPU did (job 122 cancelled before running).
2. **Per-row cost is larger than the "weights dominate" napkin says.** 16 more verify rows plus a 3rd draft step cost ~+4 ms per step
   (+14% ITL) vs my +2.6-3.1 ms estimate, cancelling the +15% acceptance. That is the same result as T1-009 on older code, so the
   FP8 draft head and DeltaLog did not change the balance.
3. With 16 running at 5.5 GiB KV (98.6% usage) there was no OOM at startup; k=3 is not "blocked", it is neutral.

## Left for others
- Freeing GDN spec-state slots under DeltaLog (T5 #2) and a cheaper per-row verify (AR/attention/GDN at 64 rows) together are what
  k=3 would need. DeltaLog on mixed-step spec rows (T5 #1, +0.5-1%).
- Review tool returned "backend error" twice (07:04-07:05 UTC): map and T7-001 unreviewed.
