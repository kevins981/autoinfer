# T16 write-up (leader-16, explore). Voluntary close after the seed

GPU: 5.0 of 20 used, 15.0 returned. Record: bank/T16-001. Worker: leader-16.w1. Notes: local_kb/leader-16.md.

## Direction tested: a hybrid decode all-reduce (copy-engine write + SM read)
- **Premise.** The decode AR is the largest bucket of the decode step still well above a floor.
  - Removing an AR outright is worth 22-26 us (T9-006, confirmed here: A_pf − N = 22.2 / 26.0 us). At 128 ARs per step that is ~2.9 ms of a ~24 ms step, about +8% e2e as a ceiling.
  - T15-002 measured a near-duplex transport floor for copy-engine D2H + SM read: 20.7 us for 384 KB, against today's ~38-46 us hostar.
- **Gate:** at least 10 us/AR faster than champion hostar + prefetch inside a real 16-layer FP8 Marlin chain (M=48, CUDA graphs, NUMA node 0 verified).

## Result (T16-001): FAILED by ~21 us/AR
- **Best arm:** raw payload, 2 copy-engine pieces, L2 prefetch blocks in the reader kernel. It is **11 us/AR slower** than hostar.
- **Standalone:** 68 us vs 38.5 us.
- **Exactness:** all hybrid arms are bitwise equal to hostar (0/300 iterations differ).
- **Cause (timeline):** inside a graph, the copy-engine branch (memcpy node + stream write-value node) takes ~20 us before its first piece lands at the peer. That is more than the whole transport-floor saving. Each extra piece adds ~4-5 us.
- **Prefetch:** the L2 prefetch is worth ~35 us/AR to the hybrid, against ~15 us/AR to hostar. Even so, it does not close the gap.
- **Compressed variant (SM encode, then CE copy):** slower still in smoke, and not completed.

## Conclusion
- Every route that puts copy-engine ops on the decode critical path pays ~20+ us of per-op latency in graphs. This holds for all-CE (T9-006) and for the hybrid (T16-001).
- At M=48 the SM-driven, compressed hostar remains the best decode transport. Its gap to the no-AR floor (~24 us) is now mostly prefetch-coupled time that no transport change recovers.

## Why I am not proposing
The other items I costed are each below +2% against the 1004.4 bar:
- GPU1 MLP rebalance in mixed steps: ~+1.7%. It also needs a memory ruling and a two-slot GDN port.
- Drafter collectives: ~+0.5%.
- lm_head screen: ~+0.8%.

Without a decode-AR gain of about 4%, a package would need roughly six such items to clear the bar. KB45 shows that stacking small exact wins tends to measure null. I agree with T2/T4/T9/T10/T15 that the champion sits close to the sum of its measured floors.

## Reusable
bank/T16-001: hyb.cu and bench.py. These are a graph-captured CE-write/SM-poll AR with per-piece write-value flags, globaltimer timelines and a bitwise checker. The record also has the measured CE-branch latency in graphs (~20 us to the first piece). Anyone considering copy engines on a latency-critical path should start from that number.
