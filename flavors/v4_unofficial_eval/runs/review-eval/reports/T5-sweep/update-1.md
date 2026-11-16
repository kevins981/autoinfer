# T5 map update 1 (resync, ~06:15 UTC)

## Tested since 6-ranking
| Candidate | Result | Evidence |
|---|---|---|
| #1 GDN DeltaLog: premise (state writes cost DRAM time in the server) | **confirmed, smaller than the napkin**. The kernel reads 26 MB and writes 37.6 MB to DRAM per launch (of ~77 MB written to L2) and takes 78.7 us. Estimated 1-write kernel: 40-64 us. | GPU job 85 (ncu, --cache-control none), agents/leader-5.w1/ncu_gdn |
| #1 kernel | built (workspace t5-dl, CUDA op + materialize op). Bitwise identical outputs and canonical states over 12 steps with random acceptance, mixed-step transitions, NULL and zero-length rows. Cold microbench 120.4 -> 61.8 us/call. | jobs 90 (test), agents/leader-5.w1/dl |
| #1 in a server, c=1 | token-identical DL on vs off, including decode across 1,600 tokens and a prefix-cache hit at a block boundary | job 92 |
| #1 in a server, c=16 | **FAILED**: GSM8K-50 30%, acceptance 2.21 (DL off: 88%, 2.36). Bisected to the log path (job 99). Likely cause: per-KV-cache-group regathering is missing for the new state-index field (review T5-2 P1). Being fixed. | jobs 91, 99, 102, 104; reviews/T5-2 |

## Verdicts changed
- 3-calibration C/D's host-bound floor for mixed steps is retracted: nsys inflated it (T4-001, review T5-1).
- The k=3 capacity claim is withdrawn: logging in slot 1 frees no allocated blocks.
- #2 (mixed-step GEMM->AR overlap) is re-priced at ~+0.5-1% (review T5-1 point 5) and handed to T6.

## New facts for the map
- The engine contains a 4th window-cost resource, host CPU issue rate. It is real (T2-001 and T3-007 were host-side wins), but it must be priced without nsys.
- The GDN layers are spread over several KV-cache groups (4 here), and every state-index tensor is per group. The graph (2-graph) should list "KV-cache group" as an index on state tensors: it is the dimension the DeltaLog bug lives in.

## Current ranking
1. DeltaLog: +2.5-3.5% (estimate), pending the c=16 correctness fix, then the harness.
2. Nothing else in T5's hands; #2 went to T6, k=3 is memory-blocked.
