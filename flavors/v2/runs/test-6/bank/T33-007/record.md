team:         T33
builds_on:    T33-004
performance:  "chain microbench (job 344): overlapped raw down kernel 52.7/51.8 vs t31b 54.1/52.4 us at PF=48 (-1.5..-2.5 us, ~noise); best overlap-only stack layer 264.0/267.1 vs champion-like ~272 (~-3%)"
summary:
  hypothesis: the overlap structure also helps down (45 MB/rank, not prefetched).
  change:     agents/leader-33.w1/ov4_down.so (3 producer + 5 consumer warps, S=8, 40 cols x 128 CTAs, 68 stages, stage-major pack); test_down.py correctness vs float64 (bf16 rounding level). Plugin ch/plug_ovd_100.py.
  results:    Observed (job 344, r0|r1 us): PF48 down 52.7|51.8 (ov4) vs 54.1|52.4 (t31b); layer 264.0|267.1 vs 266.2|269.2. PF0 down 52.5|51.7 vs 54.0|52.2. Single job, not repeated.
  conclusion: Interpretation: t31b down already runs at ~90% of the cold HBM floor (~47 us); overlap has nothing to hide there. Overlap-only stack (ov4 gate_up + ov4 down) ~-8 us/layer vs champion-like (~-0.5 ms/step, ~+1.5% e2e est.). Hand-off: agents/leader-33.w1/ASSET.md.
validity:     microbench only.
