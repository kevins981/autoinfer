# T3 write-up (sweep team, leader-3 + worker leader-3.w1)

## Results
| record | what | official | vs champion then |
|---|---|---|---|
| T3-003 | AR-window L2 prefetch of the next GEMM's weights (gate_up) by extra CTAs in the hostmem AR kernel | 897.4 | +1.97% vs T2-008 |
| T3-005 | + the next layer's in_proj (qkvz/qkv) prefetch, previously never delivered (the deferred MLP AR runs in the next norm) | 909.7 | superseded by T3-006 |
| T3-006 | T3-005's fix on T2-009 | 919.0 | +1.33% vs T2-009 |
| T3-009 | row-split AR overlap restored with T2-009's fused norm+quant (on T2-011) | **951.5** | +2.11% vs T2-011 |

Four official wins (T3-003, T3-006, T3-009 promoted; T3-005 superseded). The run went from 880.1 (T2-008, when T3
opened) to 951.5, with T2's T2-009/T2-011 in between.
Negatives and nulls: T3-001 (hostmem AR for all sizes, -6.6%), T3-004 and T3-011 (copy-engine AR, -1.56% / -2.45%
screen), T3-007 (DeltaLog L=2 ceiling; evict_first stores), T3-010 (row-split threshold 128, -1.15%), T3-012 (prefetch
56 MB), T3-013 (prefetch-aware GEMM configs), and microbenches T3-002 (CE AR) and NCCL_PROTO (job 138).
GPU: ~191 of 245 granted.

## After the first closing recommendation (sparring T3-1)
I recommended closing after update-3, sparred first (at the manager's suggestion), and withdrew the recommendation.
- T3-015 (profile, job 142): k=3's +29% step cost is mostly implementation, not inherent:
  - +4.8 ms: at 56/64 rows the GEMMs make one more M-tile pass over the weights;
  - +2.6 ms: DeltaLog caps at 2 logged rows, so 4-row steps write full states;
  - inherent part: AR +1.6 ms, draft forward +1.0 ms.
  Acceptance rises +17%. With both fixes, napkin decode step ~28.4 vs 27.7 ms. This is the run's best open lead; it was
  unfinished when the run ended (see update-4.md).
- T3-014 (microbench): 3-4 row chunks lose to 2. The token mixer's per-layer barrier costs 18-25% per layer. Removing it
  needs request-level micro-batching (vLLM's DBO excludes hybrid models).
- T3-016: row-split threshold 448 was inconclusive (-1.45%, mixed).

## What we learned
1. **An idle resource inside a window can pre-pay the next node.** The decode AR is PCIe-bound and leaves HBM idle for
   ~44 us. Prefetching ~40 MB of the next GEMM's weights there saved ~21 us per gate_up (job 88) and later per in_proj.
   The framework's Window cost max(F, BW, Link) has no term for "state of the cache after the window". We added L2
   residency as a fourth resource in our map. T2-005 had judged the idea dead from a microbench in another setting.
2. **Verify that a change actually runs.** Two of our wins came from code paths that silently didn't run:
   - the in_proj prefetch hint was cleared before the deferred AR consumed it (T3-003 to T3-005);
   - T2-009's fused norm+quant made rowsplit_ok() False, so mixed steps lost all AR overlap for three champions
     (T2-009, T3-006, T2-011). We found it in a profile (job 117) and confirmed it with one grep of server logs.
   Cheap guard: log once when an optimization path is active and grep for it in every A/B.
3. **Microbench overlap doesn't imply server overlap.** The copy-engine AR overlapped with GEMMs in isolation (T3-002)
   and lost twice in the server. Prefetch-aware GEMM configs won in a microbench and were null in the server. Both
   times, the in-situ profile, not the microbench, showed where time went.
4. **Measure the link the way the kernel uses it.** My copy-engine duplex figure (22.6 GB/s) was the wrong floor for
   the SM-issued AR; T2 measured 31-34 GB/s (their job 122).
5. **Cross-team porting is cheap and pays.** T3-006 and T3-009 were 20-63-line ports onto the other team's champion,
   each A/B'd same-job against it before the official eval.

6. **A "dead" verdict needs a decomposition, not more A/Bs.** k=3 lost three times across teams. One profile showed
   two fixable implementation step-costs (GEMM tiling and a kernel's row cap) that the per-row napkin couldn't see.
   Sparring before closing found this.

## Framework notes
- Window: add cache residency (L2) as a resource, and price the ops of a "link" node (launches, copy-engine queues),
  not only its bytes.
- Speculate: the cost of extra rows is a step function of implementation thresholds (GEMM M-tiles, kernel row
  caps), not linear in rows.
- Interactions between knob choices (a Rewrite disabling a Window choice through a guard) are not in the framework's
  per-site pricing. A regression check per choice is needed.

## Files
Map: reports/T3-sweep/1-inputs.md ... 6-ranking.md, update-1..4.md; summary reports/T3-sweep.md.
Records: bank/T3-001 ... T3-016. Sparring: sparring/ (T3-1). Scripts: agents/leader-3/scripts, agents/leader-3.w1/.
