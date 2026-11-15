# T1 write-up (sweep team, leader-1)

**Result.** BASE 635.9 -> champion T1-014 **842.7 tok/s official (+32.5%)**. T1's official wins: T1-004 (707.6),
T1-009 (786.4), T1-014 (842.7). T1-015 (row-split extended to every eager step) official 837.0: no win.
Map: `reports/T1-sweep/` files 1-6, `update-1.md`, `update-2.md`.

## What worked (in order of size)
| change | gain | record |
|---|---|---|
| TRITON_ATTN backend: in-place draft metadata, no `seq_lens.cpu()` host sync in decode | +8.4% | T1-008 -> T1-009 |
| zero-copy host-memory all-reduce (mapped pinned shm, epoch flags) at decode sizes, 43 vs 53-60 us | +4.35% | T1-002 |
| FP8 draft lm_head + local draft argmax (engine flags) | +3.1% | T1-001 |
| batch-sharded sampling (flag) | +1.8% | T1-005 |
| target-side greedy local argmax ((max, idx) per vocab shard, 8 B/row gather) | +1.2% | T1-012 |
| row-split microbatching of mixed steps (AR of one half under GEMMs of the other) | +1.1% | T1-013 |
T2's W8A8 GEMMs and cudagraph capture sizes up to 512 are in the champion too (T2-002, T2-006).

## What didn't
- Spec depth k=3: lost twice (-3.6%, -5.8%). Extra verify rows on this machine cost more than the extra accepted token
  (ridge ~80 rows for Marlin, not 333), and the TTFT p99 gate fails (T1-003, T1-010).
- AR/GEMM overlap by splitting d at decode: chunks too short, fixed costs win (T1-007).
- Reduced draft vocab: acceptance loss vs <1% saving (T1-006).
- Row-split for eager >=512-token steps: step medians -5..-17%, but official -0.7% (T1-015).
- Fused AR + add + RMSNorm: +0.26%, not significant (T1-011).
- NCCL tuning: nothing beats the defaults. FlashInfer PCIe-IPC: impossible without peer access.

## Closing tests (after spar T1-1)
- k=1: screen negative -9.2% (T1-016). With k=3's two losses, k=2 is measured optimal for this drafter.
- Marlin gap: the isolated 115 us (0.78 BW) holds with matched timing, cold weights and in a graph (T1-017). The
  in-situ 141 us is therefore caused by the serving environment, not the kernel or L2; still the largest unexplained
  excess (~+6% if closed).
- T1-015's -0.7% official is "no evidence of a win", not an established regression (between-job variance unmeasured).

## What I learned about the framework on this machine
1. **No host-CPU resource.** The largest single win (+8.4%) was a host sync in attention-metadata prep: invisible to
   FLOPs/bytes/link pricing. Price each step as max(GPU work, unoverlapped CPU prep) and look for host syncs.
2. **The link is not orthogonal.** Without P2P, inter-GPU traffic is SM-issued loads/stores to host memory, so
   overlapping the link with an all-SM GEMM mostly fails. That is also why NCCL beats the spinning host-memory AR on
   the overlapped side stream.
3. **Verification slack is not free.** The ridge is ~80 rows (Marlin bf16 compute) / ~165 (W8A8), so 48 verify rows
   already sit near it; extra draft depth costs real time.
4. **Node efficiency dominates.** The calibrated model needed measured efficiency factors (Marlin 0.66 BW in-situ vs
   0.8 isolated, NCCL at ~0.4 of link, launch floor). The gap to the speed of light is mostly in these, not in the
   knob choices.
5. Splitting on rows (not d) is the right Window for the AR || GEMM site once rows are past the ridge.

## Still open (napkin vs champion)
- Marlin in-situ vs isolated gap (~-3 ms/decode step, ~+6%): not the kernel/L2/clocks (T1-017); test the 2-GPU environment next.
- ReplaySSM (halve GDN state writes): ~+2%, high effort.
- Copy-engine all-reduce for mixed steps: ~+1%.
- Tree drafts with a tree-aware GDN verify kernel: only if per-row step cost drops.
