# local_kb leader-26 (T26 explore)
## Direction (2026-11-09): swap-AB register-direct FP8 decode GEMM (M<=48)
- Weights = MMA A operand (m16 out-features), LDG.128 global->regs from offline-permuted layout, no smem/barriers for weights.
- Tokens = n8 side (6 tiles at M=48), via smem/ldmatrix. 6 MMAs per dequantized fragment (Marlin 3).
- Differs from T8 (B via cp.async ring), T17/T24 g17 (cp.async ring, tokens on m16).
## Napkin (assumed): GDN layer in situ 354 us (T22 anatomy) -> ~299 at 88% HMMA peak w/ prefetch 30+ MB in AR2;
  -55 us/layer, ~-3.4 ms/step -> +8.8% wall; at 80%: ~-2.8 ms, +7%. Decode steps ~2250/trial.
  Pair bound: AR2+gate_up >= max(38+tensor, 89MB/0.95TB/s=94) ~105 us vs 136 now.
## Champion AR prefetch: VLLM_HOSTAR_L2PF_MAX_MB=88, 8 blocks, stride 32 B (bank/T14-004 diff).
## M0 gates: K0 no-loads <=66 us GO / >75 KILL; K1 cold >=0.93 TB/s, warm >=1.5 TB/s.
## Manager review of M0 (2026-11-09)
- Dequant in K0 is the critical cost (it doesn't co-issue with HMMA). Report regs, spills, occupancy.
- A permuted copy kept alongside Marlin's = memory change -> 128K test. Prefer one layout serving all M, or Marlin for M>48 ONLY if the same buffer is used.
- Before any ABBA: in-situ nsys check of the decode critical path (KB78 coupling).
- Messaging: use run-scoped names (test-6-leader-26.w1, test-6-manager).
## Outcome: direction KILLED (T26-001..003, ~1 GPU-min total)
- K0 swap-AB register loop 93% peak (refines KB81: dequant/fragment math is NOT the floor in isolation).
- Full kernel: load path is the bound (0.7-1.0 TB/s in-kernel vs 1.15-1.29 pure stream); tuned v3 gate_up chain 99.1 vs Marlin 94.8.
