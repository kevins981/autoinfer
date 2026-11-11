# T14 write-up (leader-14, explore)

GPU: ~46.8 of 57 granted used (seed 20, last chance 20, top-ups 9 + 8). Records: bank/T14-001..T14-004.
Worker: leader-14.w1. Notes: local_kb/leader-14.md.

## Result
- **T14-004 is the CHAMPION (promoted by the manager). Official #2: 956.6 tok/s** (trials 955.8 / 957.5). That is +0.8% over T9-005's 948.9, and both trials are above both of T9-005's.
- Gates:
  - GSM8K 94.0%;
  - p99 TTFT 1152 ms, p99 TPOT 33.1 ms: both pass;
  - acceptance 2.33;
  - 128K request passes.
- The change is T9-005 plus T6-004's Marlin policy for M>48: parallel slabs in one launch, slab-inner tile order, masked tail. It touches mixed steps only.
- Exactness:
  - decode (M<=48) is bitwise unchanged;
  - elsewhere at most 1 bf16 ulp (fp32 reduction order);
  - no memory change.
- Switch: `AI_POLICY=0` gives stock Marlin.
- Credit: the kernel is T6's (leader-6.w1). T14 ported it onto the champion, measured it in situ and landed it.
- Same-job evidence:
  - job 120 (ABBA, 300 prompts): every B above every A, +0.9%;
  - job 119 (A B B, 400 prompts): +0.2% / +0.5%.

## Forecast vs outcome
| | forecast | outcome |
|---|---|---|
| T14-1 package, payoff if it works | +5.5% (central +2%) | failed G1, never funded |
| M1 Marlin M-handling | +3.0% payoff / +1.5% central | +0.8% official |

Where the gap came from:
- My premise was wrong. I measured summed Marlin time on T2-006's T1-042 trace and got 6.9 s/trial of excess over max(0.183 ms/row, 12.7 ms).
- Re-derived on T9-005 with critical-path time (the manager's correct objection), the excess is 3.9 s:
  - The token split runs two Marlin streams at once. Each kernel slows while its twin runs, so summed time overstates the critical path by 11-15% at R>=256. This also explains most of T2's "prefill Marlin at 85% of microbench".
  - Large-M Marlin is already near the roof: 0.197 ms/row on the critical path. Cold at 64-multiples M>=192 it is 0.172-0.188.
- In situ the policy saves 0.59 s/trial of Marlin critical path (CI 0.43-0.75) and 0.9-1.3 s of mixed-step span. Almost all of that is at R<256 (T14-002).

## Measurements worth keeping
- KB-worthy: **measure Marlin in mixed steps on the critical path.** Summed kernel time over-counts the token split's concurrent streams by 11-15%.
- What is left of the mixed-step Marlin excess after the policy is ~3.4 s/trial. Most of it is at R<256: 0.252 ms/row against a 0.183 roof.
- Decode step vs number of requests (T2-006 trace): 1 req 20.3 ms, 8 req 21.5 ms, 16 req 25.5 ms. About 80% of a decode step is fixed per-step cost.
- Process notes:
  - A fresh workspace's first server start pays the JIT extension builds and a cold page cache for the weights: 309 s vs ~155 s warm. Warm both before ABBA jobs.
  - `official_eval` reserves 20 GPU-min, so the balance must cover that, not just the ~11 min actually used.
  - A refused official still creates eval-<id>-1. The retry numbering then clashed: official #1 FAILED at apply_diff, and #2 ran fine.

## Left open (none clears 5% alone)
| item | estimate | evidence |
|---|---|---|
| mixed-step glue (T2-006 list: fuse the DMA add into FusedAddRMSNorm, GDN index glue -> slices, act_and_mul, in-span idle) | <= ~1.5% | sizes from T2-006, not built |
| admission lag | <= ~1% | T10 replay: ~115 ms/request above floor; cause not measured |
| small-step Marlin, R<256, still 0.25 ms/row | <= ~1% | T14-002 |
| drain adaptive k (T5 two-slot GDN) | ~0.6% | T10 |

## Opinion
- I agree with T2-T13 that decode sits on coupled HBM, tensor and PCIe floors.
- What remains in mixed steps is a set of ~1% exact items. Each is worth landing the way T14-004 was: in-situ nsys first, then a same-job ABBA, then official.
- I found no direction with an honest payoff of 5% or more.
