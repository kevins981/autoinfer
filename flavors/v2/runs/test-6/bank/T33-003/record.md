team:         T33
builds_on:    T33-002
performance:  "microbench only: real producer->consumer smem handoff gate_up kernel (M=48, 17408x5120 e4m3): cold 100.6 us (hidden under stream), pf48 70.7 us GPU0 best (stream-only 62); missed the provisional 65 us line"
summary:
  hypothesis: W1-1's overlap survives a real handoff (cp.async ring, mbarrier/named barriers, A staged per K128, full-K CTA slabs, real bf16 output).
  change:     agents/leader-33.w1/ov2.cu (lds32 B) / ov3 (-DBLDM, ldmatrix B); 128 CTAs x 136 cols, 4 producer + 12 consumer warps (B16) or 2+6 (A8), ring S=2..4 of 23.5 KB stages; correctness vs float64 CPU reference (ref.cpp): 0 bad of 835,584.
  results:    Observed (clean jobs 312/315/316/317; GPU0/GPU1 us): best B16 4+12 S4 mbarrier ldmatrix: cold 102.6/100.4, 0.875x 90.6/88.8, pf48 70.7/65.5 (stream-only pf48 62.5/59.1). ov2 B16 S3 nb pf48 74.8/69.0. Floors: consumers-only 44-51 GPU0; L2->smem copy-only 48 us (120 MB at ~2.5 TB/s). Jobs 307/309/310 contaminated by persisting evict_last lines (discard).
  conclusion: Interpretation: cold and 0.875x are stream-bound and fully hidden (+1-3 us); at pf48 the consumer side (smem B reads, 5/4/4/4 n8 imbalance, A ldmatrix) is ~44-55 us and overlaps only partly (+9-14 us over stream-only). vs T31/Marlin at partial residency (84-90) still ~-15 us. Decisive next: in-chain AR1+gate_up pair with the champion's hint prefetch (W1-3).
validity:     microbench only.
