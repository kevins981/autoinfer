# leader-17 (T17, explore) notes
## Orientation (CPU only, 2026-11-09)
- Read T2-T16 write-ups, KB1-KB74. Floors: decode GEMM HBM/tensor coupled at M=48; AR transports all dead (CE latency in graphs ~20 us, KB74); spatial/temporal sharing zero-sum; drafter head near ceiling (KB60); FP8 MMA/INT8/FFMA not exact or not faster.
- Unbuilt gap found: at M=64 Marlin 17.65 ms vs max(mem 12.74, compute 11.7) (T6-001), 39% headroom; that cliff is what killed k=3 (T5). T8 attacked M<=48 only.
- Direction T17-1 (proposals/T17-1.md): roofline GEMM M<=64 + k=3 (two-slot GDN from T5 ws t5-rb). Payoff +6.3%, central +1.5%.
- k=3 calibration: T5-001 +3.1%, T5-004 -3.5% (noise ±1.5 per pair); model predicted ~0 with Marlin cliff -> unexplained ~-1.5%.
## Worker
- leader-17.w1 started on M0 (task agents/leader-17/w1_task.md), ws t17-k from T14-004, allowance 5 GPU-min.
- Manager approved T17-1 (grants per milestone; M1 15 on M0 pass). Conditions: M0 in a real chain w/ L2-prefetch neighbour, cold+warm; ulp check at every served M; KB61 build check; two-slot GDN port re-verified bitwise across 1600/3200 + 16-req admission + 128K; check -1.5% k=3 calibration vs T5-004 -3.5% in M2. Forwarded (1)(2) to w1.
- M0 v1 (job 127): gate_up mine vs Marlin (us) M=64 cold 134.4/103.5, chain 133.9/102.7; M=48 116.6/92.5. Exact (<=1ulp except near-zero, deterministic). Marlin numbers here lower than T5-002 (126.3/111.2) -> reconcile. Gate changed to relative: mine@64 <= Marlin@48 +5% in chain. Fit: 63 us fixed + 17.8/m-tile (no overlap).
- M0 decomposition (jobs 127,129-131): T6-001/T5-002 Marlin "cold" used a 256MB WRITE flush (dirty writebacks during GEMM, +18-20 us on gate_up). Read-flush: Marlin gate_up M=48 92.5 (= HBM roof 89MB@0.96), M=64 103.5. My premise (39% headroom at 64, 22% at 48) was an artifact of the write flush. Real headroom at 64 ~11%; at 48 ~0.
- v1 kernel M=64: math-only 88.6 us (130 TF, 89% of 146 peak); B stream alone ~0.83 TB/s; total 126-134. Floor even perfect ~97-100 vs Marlin 103.5 -> payoff ceiling ~+2-3% < bar. Awaiting job 133, then KILL.
- Job 133: B stream 0.84 TB/s, tail 6-7 us -> KILL. Filed bank/T17-001. ~0.7 GPU-min used.
