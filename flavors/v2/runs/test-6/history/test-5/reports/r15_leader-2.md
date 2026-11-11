# Round 15 report (final): leader-2 (team T1)

## Summary
No new official result this round: both experiments were null, so **T1-042 stays the final champion at 942.2 tok/s** (official, trials 941.4 / 942.9, +48.2% vs BASE). It is the record with an official result in hand. Its record is now self-contained: bank/T1-042/STACK.md, linked from record.md, covers how to reproduce, the full lineage with official numbers, every opt-out switch, the correctness summary and requirements. change.diff is the complete diff vs baseline; serve_args.txt is the full extra flags.
GPU: 16.4 / 80 used. I left the rest unspent: no other exact, already-tested lever was ready (task rule: don't start what can't reach an official eval).

## Experiments
| record | builds on | what | performance |
|---|---|---|---|
| T1-044 | T1-042 | skinny kernel for the GDN ba projection | killed before building: ba already overlaps the qkvz Marlin on a second stream; upper bound 1.8 us/layer at M=48 (line 3) |
| T1-045 | T1-042 | token-split threshold 192 / 128 vs 256 (env-only) | null: -0.5% / -0.6%, each below 256 in both pairs; keep 256 |

## Correction to round 14's anatomy (KB56)
The row "small bf16 GEMM 0.35 ms/step vs ~0.1 floor" was overlapped kernel time, not critical-path time. The priority attribution split concurrent time. The GDN ba GEMM runs concurrently with the qkvz Marlin and finishes first at 16 requests. Real exposure appears only at 1-2 requests (cuBLAS picks a slow kernel at M=3/6, ~15 us/layer on tail steps).
Lesson: per-family "gap to floor" tables from nsys must check stream overlap before a family is called a lever.

## Proposed SHARED_KB
1. KB56 correction (above) (T1-044).
2. The token-split threshold stays at 256 even with the pipelined DMA lane: 192 -0.5%, 128 -0.6% same-job (T1-045).

## What I'd do with more rounds
1. **Small-M Marlin 2-slab cliff in mixed steps** (M 105-180 reads the weights twice; ~0.22 s of a 14.1 s window, ~1.5%). A Marlin tile/schedule variant for M 65-192 that keeps one weight pass. Exact (fp reordering at most). Multi-round: Marlin internals (KB39).
2. **Exposed host-AR time in small mixed steps** (< 192 tokens, ~19 ms per step). Splitting doesn't pay (T1-045), but the copy-engine pipelining that worked for the DMA lane (T1-042) might apply to the hostar SM path at mixed sizes (0.5-2 MB), where KB5's decode-size result doesn't hold. Measure first.
3. **Tail steps at 1-2 requests**: the cuBLAS small-M kernel for the GDN ba projection (~15 us/layer exposed) and similar small-M picks. Pad to M=16 or use a tiny kernel. ~0.1% here; much more for c=1 latency.
4. **Attention**: 0.87 ms/decode step vs ~0.5 floor, 6 small kernels per layer. A fused decode-attention path for the 16 attention layers.
5. **GDN replay**: now latency-bound (first-load latency under DRAM contention, KB55). A persistent-CTA design that overlaps one request's prologue with another's stream is the remaining exact idea (~1-2 us/layer).
6. **Process**: a step anatomy that accounts for stream overlap (critical-path attribution) before choosing directions. It would have saved round 15's first direction.
