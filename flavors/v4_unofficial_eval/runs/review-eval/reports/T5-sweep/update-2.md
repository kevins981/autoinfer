# T5 map update 2 (~07:00 UTC)

## Tested since update-1
| Candidate | Result | Evidence |
|---|---|---|
| #1 GDN DeltaLog (row-0 checkpoint + log in slot 1, materialize before old consumers, off near 1600-token boundaries) | **WIN, champion**: 895.4 tok/s (898.7 / 892.1), +4.9% vs T3-007; GSM8K 93.2%, 128K pass, latency pass | bank/T5-001, job 115; review T5-2 turns 1-3 |
| mechanism | GDN decode kernel 72.7 -> 35.6 us/call (-1.79 ms per decode step); following GEMMs/AR -0.8 ms per step (less deferred L2 write-back); decode wall -8.3%; materialize +1.5 ms per mixed step | job 118 vs job 78 |
| first c=16 build | corrupted outputs (GSM8K-50 30%): materialize used one KV-cache group's block ids for all 4 groups | job 91; review T5-2 P1 |

## Verdicts changed / lessons for the framework
- **The napkin undercounted the win** (+2.5-3.5% priced, +4.9% measured): bytes written into L2 but evicted later are paid by the
  NEXT kernels. The window cost formula charges a window only for its own bytes; on a GPU with a large write-back L2 (72 MB vs
  ~77 MB written per GDN layer here) a window's writes spill into later windows. Suggested framework rule: price HBM writes at the
  point of eviction, i.e. charge (written bytes - L2 capacity that the next windows don't need) to the following windows.
- **The KV-cache group is an index on state tensors** (update-1): the bug lived in that dimension. The framework's index table
  should carry the engine's paging dimensions (group, block, slot) for kept state, not only model indices.
- Bitwise exactness made the design reviewable: every step of the change could be checked against the old op, which caught nothing
  in unit tests but made the c=16 failure unambiguous (bisect -> the integration, not the math).

## Current ranking (vs champion T5-001, 895.4)
1. L2 prefetch 24 MiB: tested, weak negative (T5-002, job 120: 889.9 vs 895.4).
2. DeltaLog for mixed-step spec rows (route them through the DL CUDA kernel; removes the 1.5 ms materialize and 2 of 3 state writes
   in mixed steps): ~+0.5-1%; needs the gate/norm gather; not started (time).
3. DeltaLog with the log OUTSIDE the spec slots + allocator change -> frees ~2.4 GB/GPU -> enables k=3 (T4's #3): the largest
   structural follow-up; hours of work.
