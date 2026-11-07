[manager, manager of run test-4] Round 2 task for leader-1 (team T1). Your own "next" (search policy rule 1).

## Direction: hide the prefill-step all-reduce behind compute
As you proposed: prefill-sized ARs are still on NCCL, link-bound, ~31% of big prefill steps and ~8% of span (KB8).
Overlap them with compute, e.g. split prefill-carrying steps into 2 micro-batches and run one's AR (copy-engine
host-staged AR on a side stream, or host AR with few SMs) during the other's GEMMs. Start with your cheapest test:
Marlin at M~1024 concurrent with a side-stream copy-engine AR, to check they don't slow each other.
Your estimate (assumed): +3-5%. Exactness: the same sums in the same precision; the split must not change the math
beyond fp reordering.

## Budget: 55 GPU-min for T1 this round.
## Champion to beat: T1-003, 691.3 tok/s official (trials 696.1 / 686.4); TTFT p99 1303, TPOT p99 46.1, GSM8K 92.8%,
AL 2.346. Start from it. Same-job A/Bs vs T1-003, repeated when close.
## Scope: team T2 works on the decode/verify path (depth-3 step cost and the ~4 ms/step inter-cluster idle in the
drafter/sampling path). Leave that to them. Memory-affecting changes need the 128K test.
Round 2 is the last round of this run (config: 2 rounds): leave time for the official eval and the report.
