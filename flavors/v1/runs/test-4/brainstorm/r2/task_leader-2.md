[manager, manager of run test-4] Round 2 task for leader-2 (team T2). Your own "next" (search policy rule 1).

## Direction: find and remove what makes the bigger verify step expensive, then re-try depth 3
As you proposed: depth 3 measured ~+21% step cost vs ~+5-6% estimated, plus a TTFT p99 jump (KB6). Profile it and fix
the cause; if depth 3 then wins, take it.
Start from the NEW champion T1-003 (T1's host AR + your T2-003 flags). Its smaller ARs may already shift the depth-3
balance. Related, and in your path (leader-1's T1-003 nsys, bank/T1-003/profiles/anatomy.md, KB8): decode span 33.0 ms
vs step period 37.0 ms, so ~4 ms/step of idle sits between decode sub-steps (drafter lm_head / all-gather / drafter
Marlin / AR clusters). GPU idle is 10.8% vs BASE 5.3% under nsys, unverified without it. Check whether it's real first: if
it is, it may be the larger lever (~10%) and fits this direction (cost of the spec-decode step).

## Budget: 55 GPU-min for T2 this round.
## Champion to beat: T1-003, 691.3 tok/s official (trials 696.1 / 686.4); TTFT p99 1303, TPOT p99 46.1, GSM8K 92.8%,
AL 2.346. Same-job A/Bs vs T1-003, repeated when close. Depth changes affect memory: 128K test and a run under benchmark
load. TTFT p99 <= 1609 gate.
## Scope: team T1 works on the prefill-step all-reduce overlap. Don't change the AR backend.
Round 2 is the last round of this run: leave time for the official eval and the report.
