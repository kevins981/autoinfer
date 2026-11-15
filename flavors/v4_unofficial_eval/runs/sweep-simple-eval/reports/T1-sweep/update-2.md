# T1 map update 2

## Tested since update-1
| candidate | record / job | result | verdict |
|---|---|---|---|
| W-DBO in mixed steps | T1-004, job 18 (microbench) | overlap of NCCL AR with a compute-bound GEMM hides only ~50% of the AR; half-row GEMMs cost +25% per row: net ~0 | **dropped**. The framework's `max(F, BW, Link)` window cost is optimistic here: NCCL uses SMs and the host path, so Link and F are not orthogonal (07 §2.3 interference, measured: ~50%) |
| NCCL tuning on Simple (channels, threads, buffers) | T1-005 (microbench) | all within ~9 us repeat noise at 48 / 260 rows | dropped |
| NCCL_PROTO=Simple (I-AR a) | T1-002, **reverted** | NCCL debug log (job 33): Simple *is* applied; but in-engine NCCL time is unchanged (decode @48 rows 9.92 vs 9.66 ms T2-001; prefill fit 38.4 vs 33.5 ms @260): the +1.0% unofficial was noise | **dropped**. Lessons: (1) NCCL kernel names (`_RING_LL`) do not show the protocol; use NCCL_DEBUG=INFO SUBSYS=TUNING; (2) standalone collective microbenchmarks did not transfer to the engine; (3) check the in-engine mechanism before calling a candidate |
| host-memory custom AR (I-AR b) | T1-003 | exact, standalone not faster than NCCL | parked |
| **R-ARGMAX target side** | **T1-006, promoted, official 712.1 tok/s (+12.0% vs BASE, with T2-001/T2-004)** | unofficial +2.5% vs T2-004; profile: logits AllGather 0.934 -> 0.053 ms/decode step, full-logits sampling kernels -> one 2 us greedy-verify kernel | **win, confirmed**. Napkin (~0.9-1.0 ms/step, ~+2%) matched the measurement |
| R-ARGMAX on T2-005 | T1-007 = T2-006 (identical) | T2-006 promoted 738.8 / 747.8 | duplicate; eval cancelled |
| MTP k=3 on BASE | T1-001 | 602.5 (1 start), acceptance 2.70, step +22% | retest on the cheaper-draft champion (next) |

## What the framework got right / wrong in this run
- Right: the vocab dimension as a split index with an argmax-merge combine (Rewrite: push the reduction below the gather) was a clean, priced, exact win: the napkin predicted it within ~20%.
- Wrong / missing: (1) the Link resource for a no-P2P pair is two PCIe hops through host memory: ~13 GB/s of all-reduce payload, not 25; (2) concurrency of comm and compute is far from `max()` on this GPU (~50% overlap); (3) a Host resource (T2's addition) was needed to explain mixed steps.

## Current ranking (T1 lane, vs T2-006 ~743 unofficial)
1. MTP k=3 on T2-006: drafts now cost ~1/3 of before (reduced-vocab head + local argmax + target argmax); acceptance 2.70 vs 2.31: +17% tokens/step; step cost napkin now ~+10-15% -> 0..+5%; latency-gate risk (TPOT p99). One start.
2. Communication: the decode all-reduce floor (~57 us x 135 = ~7.7 ms/step) remains the biggest non-GEMM cost, but every transport tried (NCCL variants, host-memory custom, flashinfer PCIe IPC) is at the floor or worse. Only structural options remain (fewer ARs per step), none priced positive at TP=2.
3. Draft-side remaining: second draft step's AR + MTP layer (~0.5 ms) — small.
