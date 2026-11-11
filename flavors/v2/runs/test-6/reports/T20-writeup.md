# T20 write-up (leader-20, explore). Closing voluntarily: G1 failed

GPU: 3.8 of 20 (seed) used. Record: bank/T20-001 (worker leader-20.w1). Proposal: proposals/T20-1.md (approved; the start grant was conditional on G1, never released). Notes: local_kb/leader-20.md.

## Direction
A package forecast at +5.9% payoff if it works, central +2.7%. Its core item was a fence-free, flag-in-data ("LL") host all-reduce for decode:
- the champion's 0.781 codec bits packed into 63-bit payload words, with a 1-bit epoch tag per 8 B word;
- the reader streams words as they land, with no system fence and no flag.
The other items were a GPU0/GPU1 rebalance (uneven MLP split, decode compensated through the lm_head vocab split), drain adaptive k, and mixed-step glue. Together they are ~3.1%, below the bar without the AR item.

## Measured (T20-001)
- **Atomicity holds in practice.** 8 B tagged units with 16 B vector stores and loads, over host memory without P2P: 0 mismatches.
  - 104,928 ARs and 5.1e9 wire words, checked bitwise against hostar;
  - arms: rank 0 late, rank 1 late, and both jittered, each with an HBM-heavy neighbour.
  - The negative control detects stale words (tag off: 60/60 differ; tag on: 0/60).
  - This answers T18's objection for the 8 B, NCCL-LL assumption. Wire overhead is only +2% bytes.
- **Standalone:** 38.4 -> 33.1 us/AR. The writer needs 14-15 us to issue its stores, so it sits 7 us above the 25 us SM W∥R floor.
- **In a 16-layer real Marlin chain with prefetch:** only +1.9..+3.2 us/AR saved; the gate was >= 6.
- **Mechanism:** in the chain, the decode AR kernel lives ~69 us because its L2-prefetch blocks stream the next GEMM's weights. The transport itself is done by ~33 us.
  - A faster transport shortens the prefetch window, and the prefetch credit (KB41) gives most of the saving back.
  - T16 saw the same effect.
- **Ceiling:** even a floor-level 25 us AR would save only ~6-7 us/AR in the chain, about +2% end to end.
- CPU-only, on T14's in-situ nsys (agents/leader-20/idle.py): decode steps have no hidden GPU idle. The large kernel gaps (~4% of the window) are copy-engine AR waits in mixed steps, already in T19's exposed comm.

## KB candidate
KB78. Fence-free tagged host AR: SM-only, 1 epoch bit per 8 B word, champion codec bit-packed (+2% bytes). It is bitwise and survived 1.05e5 skewed, HBM-loaded ARs (5.1e9 words) with 0 errors, and a negative control proves stale-word detection. Standalone 38.4 -> 33.1 us at M=48, but only 2-3 us/AR in a real chain: the decode AR's in-chain duration is set by the L2 prefetch (~69 us), not transport. Faster decode AR transport is worth <= ~6-7 us/AR in situ even at the 25 us floor (~+2%). (T20-001)

## Opinion
Decode AR transport is now closed from both sides:
- copy-engine paths lose on start latency in graphs (KB74);
- the SM path reaches within 8 us of the floor, but the prefetch credit absorbs the saving.

What remains are the exploit-sized items listed by T14, T18 and T19 (rebalance, small-step Marlin, glue, drain k), each ~1-2%. They are best harvested one by one in situ by an exploit team. If anyone takes the rebalance: it can be done without duplicated weights (uneven MLP split plus a decode-only lm_head vocab shift toward GPU0). Memory then moves between ranks, so it needs the 128K test.
