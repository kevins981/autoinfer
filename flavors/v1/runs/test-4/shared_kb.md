# SHARED_KB (run test-4). Confirmed understanding; each entry cites its records.

KB1. All-reduce on this box (no P2P): NCCL RING_LL is the default and NCCL env knobs (protocol, channels) never beat it, at
decode sizes (T<=96) or prefill sizes (T=256-2048). (bs-r1-adjacent_field jobs; T1-001 record)
KB2. Host-staged 2-rank AR (pinned shared host memory, one-shot, fp32 add -> bf16) is bitwise equal to NCCL and beats it at
small sizes: T=16 ~20-25 vs 43 us, T=48 45 vs 68 us, T=96 80-84 vs 100 us; ties at T>=256. In-server mean 44.5 vs 61.4 us,
~2.2 ms saved per decode step; +5.7% official alone. (T1-001)
KB3. SM zero-copy PCIe is not full-duplex here: 491 KB write 18.7 us, read 21.5 us, concurrent 29-33 us; fence.sys + flag
round trip ~2.5 us. T=48 AR floor ~34 us; the host-AR kernel is within ~10 us of it. Further AR-kernel tuning: small upside. (T1-002)
KB4. FP8 (W8A8) draft lm_head + local-argmax drafting leave MTP acceptance unchanged (~2.35). --enable-batch-sharded-sampling
works with MTP on the V2 runner and is exact. Together +2.8% official. (T2-001, T2-003)
KB5. A custom exact greedy vocab-parallel target argmax adds nothing on top of batch-sharded sampling: the target-logits
exchange is no longer a lever. (T2-004)
KB6. Draft depth 3 (on T2-001): acceptance 2.35 -> 2.76 (position 3: 0.41) but ~+21% step cost (est. +5-6%), -3.9% tok/s,
TTFT p99 2225 ms (fails the gate). Cause not profiled. (T2-002)
KB7. Host AR and the T2 flags stack nearly additively: +5.7% and +2.8% alone, +8.7% together (official). (T1-003)
KB8. T1-003 anatomy (nsys, GPU 0): Marlin 51% of span, all ARs 20.4%. Decode span 33.0 ms but step period 37.0 ms:
~4 ms/step of idle between decode sub-steps (drafter/sampling clusters), GPU idle 10.8% (BASE 5.3%) under nsys (could be
inflated by nsys). Prefill steps unchanged: NCCL AR ~31% of big prefill steps, 8% of span; large-AR skew p50 ~196 us.
(bank/T1-003/profiles/anatomy.md)
KB9. Short-run p99 TPOT swings 42-57 ms between two benches on one server: short runs can't judge the latency gate. (T2-003)
KB10. Baseline step anatomy (nsys BASE, GPU 0): decode step ~35-40 ms = Marlin ~17.5-18.6, AR ~6-8, lm_heads ~4.3
(target 1.59 + drafter 2 x 1.37), logits all-gathers ~1.7, GDN decode ~2.4-3.7. AR 23% of wall; prefill-containing steps
36% of wall. PP=2 ruled out by napkin math (doubles weight streaming). (brainstorm/r1/*.md)
