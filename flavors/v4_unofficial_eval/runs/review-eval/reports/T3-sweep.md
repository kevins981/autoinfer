# T3 sweep (leader-3): summary
Map: [1-inputs](T3-sweep/1-inputs.md), [2-graph](T3-sweep/2-graph.md), [3-calibration](T3-sweep/3-calibration.md),
[4-knobs](T3-sweep/4-knobs.md), [5-compare](T3-sweep/5-compare.md), [6-ranking](T3-sweep/6-ranking.md). Review: reviews/T3-1 (answers below; corrections
will go into update-1).

## Review T3-1: my answers
1. in_proj_ba overlaps qkvz on a side stream (s17): ADOPTED, confirmed by the worker's microbench (F.linear 5.8 us isolated) and the
   timeline; candidate #1 dropped.
2. Mixed-step skew estimate compared different sizes: ADOPTED; #5 unpriced until size-matched waits are measured (deprioritized; T2 measured
   ~7.65 ms GPU idle per mixed step with TRACE=graph).
3. Kernel sum vs union (1.18 ms overlap; idle 2.05 ms): ADOPTED.
4. Mixed GEMM rate (~205 TFLOP/s achieved; double-counted decode tokens in mixed steps): ADOPTED; the whole-benchmark napkin overestimate is
   largely my double count, not nsys.
5. GDN writes really are 3 slots; the write-skip test as designed would break source selection: ADOPTED. The worker's cold microbench
   (separate test, sources kept) shows 120/89/60 us for 3/2/1 writes at ~845 GB/s; real-server kernel 70 us -> part of the write traffic is
   attributed elsewhere (write-back). DeltaLog stays open, unranked until a dram-bytes check.
6. lm_head: agreed.
7. DBO blocked (hybrid unsupported, mamba asserts ubatch 0, HMAR disabled under ubatching): ADOPTED; dropped.
8. SendRecv = vocab-shard all-to-all of target logits, ~6 MB/rank: ADOPTED; promoted to a build candidate (exact greedy fast path: local
   (max,idx) per shard, tiny gather, greedy verify; fallback for any non-greedy feature).
Other: reduced-vocab drafting "unsettled on this champion" (agree, but memory headroom + T1-002's acceptance loss keep it low);
PP bytes (agree: 12.17 GB per stage; PP still loses); tree pruning reason -> implementation cost (agree); chunk size not only AR (agree,
unpriced). Column-split (#3): worker emulation (task 1, D) shows no gain (97.3 serial vs 97.0 at best) -> dropped.

## Current ranking (after review + microbenchmarks)
1. L2 weight prefetch under each hmar all-reduce (worker microbench: -10 us per qkvz GEMM, -13 us per gate_up GEMM under a 40 us spin
   -> ~1.5 ms/step, ~+3%) — building (worker, on T2-007).
2. Exact greedy target fast path replacing the logits all-to-all (~0.45 ms/decode step + mixed: ~+1-1.5%) — next.
3. GDN DeltaLog — needs a dram-bytes check first.
