# T3 map update 1 (after 6-ranking)

## Tested since 6-ranking
| candidate (rank) | record | result | evidence |
|---|---|---|---|
| 1. AR-window L2 prefetch (gate_up) | T3-003 | **official 897.4 tok/s, +1.97% over T2-008: champion** | ABBA +2.41% (job 85); official job 87 |
| 1b. same, + the next in_proj (qkvz/qkv) prefetch before the deferred MLP AR | T3-005 | ABBA mixed +0.85% (job 101); **official 909.7 tok/s, +1.37% over T3-003** | jobs 88-101 |
| 2. copy-engine AR, microbench | T3-002 | overlaps with GEMMs (CE‖gemm = gemm +3-14 us vs NCCL +27-115, hostmem +87-333) | job 84 |
| 2. copy-engine AR in the server row-split | T3-004 | **-1.56%, all B < all A** | job 86 |

## What the measurements changed in the map
- **The Window knob's max(F, BW, Link) needs a fourth resource: L2 residency.** An HBM-idle window (the PCIe-bound
  AR) can pre-pay the next node's weight read. In situ (job 88), gate_up dropped 21-22 us (113->92 after attention,
  132->111 after GDN), and the AR got 2 us slower. That is ~1.3 ms/decode step, against a 1.5-3 ms napkin. The
  in_proj half of the napkin was at first silently not running, because of a code-path detail (deferred all-reduce).
  Lesson: verify that a hint is consumed before measuring.
- T2-005 found no gain from an L2 prefetch microbench. It was run before DeltaLog and as a separate kernel. Fused into
  the AR kernel and run in situ, it does gain. The verdict "dead" was wrong because of the setting, not the idea.
- **The copy-engine AR's microbench win doesn't transfer to the server** (T3-004). My working hypothesis is that
  mixed-step exposure is dominated by rank skew and waiting, not by SM contention. That matches T2's observation
  (half of the mixed-step AR time is waiting for the peer). Rank 2 drops to "needs a profile to explain"; it is no
  longer a candidate on its own.

## Current ranking
1. (done) T3-005: official 909.7, champion candidate.
2. DeltaLog with a 2-step checkpoint (~1.5% napkin): untried, medium effort.
3. Rank-skew reduction in mixed steps (the real limit behind T3-004?): needs a profile first (~4 GPU-min).
4. Prefetch tuning: MB, the target ranges (e.g. the rest of gate_up beyond 40 MB, the down weight), and more blocks
   in steps with longer ARs. Small (<0.5%) each.
5. Reduced-vocab draft head with a generic frequency table: marginal.
