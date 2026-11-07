# SHARED_KB (run test-3)

KB1. On BASE, prefill-containing steps are GPU-bound, not CPU-launch-bound: graphing them (prefill graph cap 2048) gives only +2.2% unofficial on NCCL. (T1-001)
KB2. In-server decode AR: NCCL 60.8 us -> host-staged 49.6 us mean (post-arrival 48.1 -> 34.0 us), ~1.5 ms/decode step saved; all of T1-002's gain is from decode. (T1-002, T1-003)
KB3. Prefill-step ARs (~620 us) carry ~250-316 us mean rank-start skew; transport speed doesn't change them (post-arrival ~510-520 us with either backend). (T1-003)
KB4. Prefill-graph flag on top of host-staged AR cuts skew (prefill AR 316 -> 214 us, decode 30 -> 10 us) and pays +4.8% vs +2.2% on NCCL: the two stack super-additively. Cause not separated (skew vs launch count). (T1-005)
KB5. PCIe here only partly overlaps up/down traffic from SMs: concurrent host write+read of 491 KB = 41 us vs 25-27 us alone; a system-scope fence/ack round trip ~5 us. Host-staged AR at M=48 is within ~5 us of its floor; chunk pipelining loses. (T1-003)
KB6. Logits all-gathers ~1.1 ms/decode step (946 us target + 2 x 330 us drafts, partly skew); the host path saves only ~0.3 ms. (T1-004)
KB7. CUDA-graph capture memory with host-staged comms: 1.33 GiB/rank vs 2.07 GiB with NCCL at the same sizes. (T1-001, T1-005)
