team:         T3
builds_on:    T3-009 (microbench only)
performance:  microbench: 3-4 row chunks lose to 2; the per-layer barrier (token mixer) costs 18-25% per layer at N=2
summary:
  hypothesis: (from spar T3-1) mixed-step AR is still ~7% of wall exposed; splitting into N=3-4 row chunks shrinks the
              1/N pipeline bubble (napkin +1.5-2% of wall).
  change:     nchunk_bench.py: 2 ranks, 8 layers x per chunk [gate_up + down W8A8 GEMMs at M=R/N, NCCL AR of the chunk
              on a side stream]; variant barrier=1 waits for all chunks' ARs before each layer (models the token mixer
              needing all rows). us per layer, job 141.
  results:    Observed (us/layer; N=1 serial):
              R     N=1    N=2 (no barrier / barrier)  N=3            N=4
              512   939    718 / 878                   814 / 941      1024 / 1064
              1024  1785   1231 / 1540                 1387 / 1605    1416 / 1570
              1536  2673   1779 / 2226                 1827 / 2140    1867 / 2107
              256   543    722 / 766                   1047 / 1197    1457 / 1484   <- at 256 rows splitting LOSES (+41%)
  conclusion: Interpretation: more chunks don't pay (NCCL fixed cost per AR + smaller GEMMs); the real limit is the
              per-layer barrier: barrier vs none costs 18-25% per layer at N=2. Removing it needs the token mixer split by
              request (micro-batching with per-half attention/GDN metadata). vLLM's DBO excludes hybrid models and only
              yields at MoE comm points, so that's a new build: napkin up to ~3% of wall; large effort. Not started.
validity:     microbench only.
