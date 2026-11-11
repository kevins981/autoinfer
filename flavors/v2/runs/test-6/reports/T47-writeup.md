# T47 write-up (leader-47, research). Closed after stage 1 of T47-1 + one measurement

GPU: 5.8 of 20 (seed) used; jobs 423, 427 (T47-001), 428, 429 (T47-002); worker leader-47.w1. Proposal: proposals/T47-1.md.
Sparring: sparring/ T47-1. Notes: local_kb/leader-47.md. Analysis scripts: agents/leader-47/ana/.

## T47-1: asymmetric waves (decode graph as cover for the prefill's all-reduce) -- gate FAILED (bank/T47-001, KB128)
- Measured (host-free replay, T45-005, 60 decode rows): J - P||D = -4.9 / -4.1 / +0.65 / +2.7 ms at P = 64/128/256/512 (gate +8 at 256).
- Why my model was wrong: decode rows inside a joined mixed step cost only J - P = 10-16 ms (alone 22-29 ms): the shared weight
  stream already halves their price. A split must hide > (J-P)/D ~ 0.5 of the decode wave; measured overlap 0.43-0.54.
- Two copy-engine-AR waves do not overlap at all (suspected CE queueing, unverified). One graph with two branches never overlaps;
  separately launched graphs do; stream priority is irrelevant.

## T47-002: DRAM bandwidth does not need all SMs (new fact, corrects the reading of KB59)
- Streaming read: 950-957 GB/s from 16 SMs (cp.async ring 48-96 KB/SM) or 32 SMs (ldg); the 128-SM best is no higher.
- Today's decode GEMMs slow in steps when their grid stops fitting one wave (qkvz t31b at 112 SMs, down at 64, Marlin at 112/48);
  the champion gate_up path holds within +20% down to 48 SMs. At 128 SMs they run at 1.32-1.41x their cold byte floors.
- Interpretation: the "zero-sum" of spatial sharing (KB59, T2) is a property of today's kernels (grid/wave sizing, per-SM
  pipelines), not of DRAM. T4's AR-beside-GEMM interference (memory-system queueing of host round trips) is a separate effect
  and stays.

## Other findings (CPU)
- Draft level on T37-002: 0.54 ms (level 1 0.66 at 64 rows), drafting ~2.0 ms/step; ~0.15 ms/level above a realistic floor (KB80 holds).
- KV/state pool (5.0 GiB) is ~2x oversized for either the 128K gate or 16 requests at k=3: 1-2 GiB reclaimable; the memory veto
  on replicated out_proj (T45-002) is implementation. FP8 all-gather of the out_proj input (3.2 vs ~8 KB/row) with a replicated
  out_proj: decode ~wash, prefill ~+2.5% (napkin).
- GDN draft trees need no extra state reads (branch state = scalar x shared state + low-rank update), but sibling rows yield
  ~0.08 tokens vs a ~0.16-token row price: chains dominate at this acceptance.

## Open questions I'd take next
1. Decode GEMMs sized for 32-64 SMs (persistent grid, deep cp.async ring), measured in-chain at their byte floor. If they hold,
   re-test T2's spatial prefill lane and decode/prefill co-running on them: the free SMs become a resource.
2. Why two copy-engine-AR waves serialize (per-engine trace).
3. Greedy-token check of the decode-row diff in T47-001 (0.16-0.24 rel-norm at the final layer) before any wave design is used in-server.
