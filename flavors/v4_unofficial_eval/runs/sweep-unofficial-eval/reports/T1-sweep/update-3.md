# T1 map update 3 (HEAD = T1-012 after promotion; official eval of the chain running)

## Tested since update-2
| candidate | record / job | result | verdict |
|---|---|---|---|
| CPU sync in attention metadata (new site: the K/R knob on *host-side* metadata, absent from my first map) | T1-010; jobs 65, 76, 82, 84 | TRACE=graph: decode-step idle 2.56 -> 0.14 ms; ABBA +1.1%; unofficial PASS +1.89% vs T2-012 | **won (HEAD)** |
| Mixed-step idle (7.5 ms/step in nsys) | t1-w1 task 12, GPU-free | thousands of 1-13 us launch gaps in eager GDN prefill chain; no single sync >= 2 ms; T2-014 cut CPU launch cost 1/3 with 0 end-to-end change | closed (nsys overstates CPU gaps ~10x) |
| Decode budget vs floors (T1-010) | t1-w1 task 13 (agents/t1-w1/budget/budget.md) | step 29.55 ms: Marlin 17.6 (floor 12.1), host AR 5.6 (at floor), GDN 3.7 (2.4), lm_head+exchange 2.1 (1.6), drafter 1.7 (~1.4 incl. AR), idle 0.14 | map recalibrated |
| gate_up slower in GDN layers (138.6 vs 117.0 us in situ; 104-111 isolated) | jobs 89, 90, 91 | not L2 contents, not preceding kernels (real GDN/AR/norm in graph), not clocks, not smem carveout, not TLB (48 buffers, scattered slots) | unexplained; parked (~1.5 ms/step prize if found) |
| W1b: cap max_num_batched_tokens 512 (T2's idea) | T1-011 | -1.05% (2 starts), TTFT p99 +116 ms | dropped: big prefill steps are GPU-bound |
| Chunked GEMM+AR overlap (W/Split of AR edge), revisited | T1-012 | PASS +0.52% vs T1-010; per-layer fp-reorder-exact | candidate |

## Framework notes
- My phase-1 map priced FLOPs/bytes/link; it had no row for **host (CPU) critical path** sites: a CPU-blocking D2H in metadata
  build was worth ~2% and was invisible to napkin math. The graph should include the host scheduler/metadata nodes, with
  their sync edges, as compute nodes on a third device (the CPU).
- nsys per-kernel traces inflate CPU launch gaps; T1-010's profile gain (-11% step) became +1.1-1.9% real. Price CPU-side
  candidates from unprofiled step times.
- In-situ GEMM time is 1.46x the cold-HBM floor, while microbenchmarks reach 1.07-1.14x. That interaction is not in any napkin
  model I have, and a microbench doesn't reproduce it.

## Ranking now (against HEAD T1-010, or T1-012 if promoted)
1. (open) Find the in-situ GEMM excess (gate_up/out_proj/down in GDN layers, ~2 ms/step): next test would be a real 2-layer
   run from the loaded model (several hours of work). Uncertain.
2. AR+norm fusion at grid 16 (T1-009) rebased: +0.28% screen; <=0.5%, cheap unofficial eval.
3. Draft lm_head/eh_proj to floor: ~0.13-0.3 ms/step (<1%), kernel work.
Everything else in my map is built, tested negative, or owned by T2 (GDN state, KR1).

Note: T1-012 turns off use_fused_allreduce_norm (the AR deferral into the next norm) whenever chunking is enabled, so
T1-009 (fused AR+norm in the host AR) can't stack on it without per-step gating of the deferral. Its value (+0.28% screen) doesn't
justify that work.
