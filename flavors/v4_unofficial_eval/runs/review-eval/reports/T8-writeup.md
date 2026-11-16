# T8 write-up (leader-8, sweep team; no worker started)
**Wins: none.** Record: bank/T8-001 (fp16 accumulation in W8A8: microbench only, ruled not allowed). GPU used: 0.2 of 45 GPU-min (job 124).
Map: reports/T8-sweep.md (files 1-6, ranking with an evidence addendum).

## What I learned
1. The framework's sweep reproduced the known cheap candidates (k=3, L2 prefetch size), but the history had already tested them all
   (T7-001, T1-009, T5-002). With a late start, phase 2 is where the value is.
2. A rule-text dimension (accumulator precision inside a K-block) is a real implementation knob the framework doesn't represent.
   It is precision, so it is a contract question first: ruled not allowed. Measured: 1.09-1.23x best-vs-best, not 2x.
3. The 4090 runs fp32-accumulate FP8 at 240-275 TFLOP/s in Triton (job 124), above SPEED_OF_LIGHT's 165, which shifts prefill
   napkins: mixed steps are not compute-bound. In the champion's profile, ~65 ms of a ~120 ms mixed step is NCCL AR wait plus GPU idle.

- Job 125 (bank/T8-002): mixed steps have a ~110 ms floor independent of prefill size; at small prefills the GPU idles ~60 ms of it.

## For the next team
- Top lever: the mixed-step idle/AR wait (6-ranking #1 + addendum). Start with a TRACE=graph profile of mixed steps on T5-001.
- Second: a multi-step GDN state log (write the state every 2nd decode step), building on T5's DeltaLog.
