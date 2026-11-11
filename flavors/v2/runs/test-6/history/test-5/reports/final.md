# Final report (run test-5, resumed from test-3; rounds 1-15)

## Result
- **CHAMPION: T1-042, 942.2 tok/s official** at c=16 (trials 941.4 / 942.9), **+48.2% vs BASE 631.4**.
  GSM8K 93.2% (gate 89.6), TTFT p99 1100 ms (gate 1784), TPOT p99 35.5 ms (gate 58.1), MTP acceptance 2.325, 128K request pass.
- Reproduce: bank/T1-042/STACK.md (lineage, switches and defaults, correctness summary, requirements); change.diff = full diff vs the task baseline; serve_args.txt = all extra flags.
- For comparison: the independent prior/baseline-run-1 reached 748.0 mean. Its untried wins were ported in round 4 (TRITON_ATTN, draft vocab subset, Marlin decode tiling).

## Champion path (official tok/s)
BASE 631.4 → T1-005 699.2 (r1: host-staged AR + prefill graphs) → T1-009 730.9 (r2: logits diet, FP8 draft head, greedy local argmax) → T1-012 749.7 (r3: GDN spec-state replay) → T1-016 805.7 (r4: TRITON_ATTN + draft vocab subset + Marlin tiling, from run-1) → T1-020 850.3 (r5: L2 weight prefetch during the decode AR) → T1-025 874.8 (r7: GDN replay on mixed steps, every-32 graph grid, DMA/hostar token-split overlap of prefill ARs) → T1-026 880.9 (r8: NUMA self-bind; fixed a per-start bimodality) → T1-030 897.3 (r10: lossless AR payload compression) → T1-037 922.8 (r12: lazy GDN checkpoint + losslessly encoded target lm_head with a custom GEMM) → T1-040 929.1 (r13: 8-lane GDN reductions) → T1-042 942.2 (r14: GDN load scheduling + pipelined copy-engine AR).
Rounds 6, 9, 11 and 15 produced no gain (closed decode Marlin tuning/rewrite, small-win stacking, the ba GEMM). Full tree: graph.md.

## What was learned (shared_kb.md, KB1-KB58; highlights)
- The hardware: 2x 4090 without P2P; GPU 0's SM clock is locked (KB8); both GPUs on NUMA node 0, and pinned host buffers must live there (KB37).
- Decode step (~25.5 ms): Marlin near the HBM roof cold and without ALU slack warm (KB28, KB39, KB49); host AR at the PCIe floor, made cheaper only by moving fewer bytes (lossless compression, KB43) and by filling its window with L2 prefetch (KB25); GDN went from HBM-traffic-bound (replay, lazy checkpoint) to shuffle-bound to latency-bound (KB48 corrected by KB51, KB54).
- Mixed steps (~39% of wall): prefill ARs hidden by a token split on the copy engines, which pipeline well (KB38, KB55).
- Corrections made during the run: KB35 → KB38 (slow DMA copies were a job-wide slowdown), KB48 → KB51 (GDN not HBM-bound), KB56 → KB58 (overlapped kernel time is not critical path).
- Process: compare only within one clean job (no concurrent builds, KB53); negative controls must delay the producer before its copy (KB57); a lib.sh harness bug was found and fixed by the human (round 10).

## Rulings made in this run
- Lossless re-encoding of weight storage counts as "weights stay FP8" (manager, round 12; same logic as round 3's lossless AR ruling). Used by T1-037 (lm_head). The human may override.

## Left open (leader-2's list, reports/r15_leader-2.md; untried ideas: ideas.md)
1. Small-M Marlin "2-slab" cliff in mixed steps (M 105-180 reads the weights twice; ~1.5%), multi-round.
2. Copy-engine pipelining for the hostar SM path at mixed sizes (0.5-2 MB); measure first.
3. Tail steps at 1-2 requests: small-M cuBLAS picks (~0.1% here, more for c=1 latency).
4. Fused decode attention (0.87 ms/step vs ~0.5 floor).
5. Persistent-CTA GDN replay (~1-2 us/layer).
6. Critical-path (overlap-aware) step anatomy before choosing directions.
- Archived with a ruling needed: R12-C, a certified FP8 prefilter for the greedy argmax (extra FP8 copy of a target weight; +636 MB/rank).

## Process notes
- One leader for most of the run (leader-2, rounds 4-15). Search policy rule 1 skipped the brainstorm in 8 of 12 rounds; brainstorms ran in rounds 7, 10 and 12 (after null rounds), and each led to a winning direction.
- The manager session was restarted 6 times by the background-session idle limit (journal); no state was lost.
- test-3's final report (rounds 1-3) is kept as reports/final_test-3.md.
