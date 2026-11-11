# T26 write-up (leader-26, explore). Recommending a voluntary close after the seed

GPU: 0.5 of 20 used (worker leader-26.w1, jobs 165-167 + W1-3). Records: bank/T26-001..003. Notes: local_kb/leader-26.md.

## Direction tried: a "swap-AB, register-direct" FP8 decode GEMM at M=48
- **Premise.** T8, T17 and T24 all had three things in common:
  - they staged weights through shared memory (a cp.async ring with a barrier per stage);
  - they put the tokens on the MMA's m16 side;
  - their loops topped out at 73-79% of HMMA peak.
- **The new design point:**
  - weights on the MMA A side, loaded global -> registers with LDG.128 from an offline-permuted layout;
  - no shared memory or barriers on the weight path;
  - tokens on n8, so each dequantized fragment feeds 6 MMAs instead of 3.
- **Forecast (assumed, from T22's in-situ per-layer anatomy):** the decode step would drop from 25.3 to ~22 ms, a payoff if it works of ~+7-9%. Odds ~15%.

## Measured
| step | result |
|---|---|
| K0 loop only (T26-001) | 62.9-64.1 us on gate_up-sized work = **92-93% of HMMA peak**, with Marlin-identical dequant + scale. Dequant costs ~1 us. Flat across 8-16 warps/SM and 1-4 tiles/warp. |
| K1 stream only (T26-001) | cold 0.95 TB/s (HBM roof); warm qkvz 1.29 TB/s, out_proj 3.7 TB/s |
| v1 full kernel (T26-002) | Slower than Marlin on every shape. gate_up chain P=32: 106.8 vs 94.4 us. qkvz: 64 vs 43.8 us. |
| v3 tuned (T26-003) | 8 warps/SM, no spills, 8-deep register ring, balanced split-K. Still slower: gate_up chain 99.1 vs 94.8 us; qkvz 57.8 vs 43.3 us. The kill line (gate_up <= 85 us or qkvz <= 39 us) was missed. |

All versions are bitwise identical to Marlin in dequant and differ only by fp32 summation order (max |d| <= 4.9e-4). They are deterministic.

## KB candidate (refines KB81)
- **What K0 shows.** At M=48, a swap-AB inner loop with Marlin-identical FP8 dequant reaches 93% of HMMA peak when weights come from registers. So dequant + MMA issue is not the per-fragment floor that KB81 inferred.
- **What the real kernel shows.** Inside a real kernel, the weight stream falls to 0.7-1.0 TB/s, against 1.15-1.29 TB/s for the same stream run alone. That makes the load path the bound, and Marlin stays ~5% ahead in the chain arm (gate_up) and ~33% ahead on qkvz.
- **Suspects, not isolated:**
  - per-CTA token staging (~63 MB of L2->SM traffic per GEMM);
  - barrier lockstep;
  - scattered per-warp weight streams.
- **Implication.** A decode GEMM that beats Marlin needs a token path that doesn't re-read the activations once per CTA, for example tokens kept in registers across tiles or broadcast. Even then, the cold bound is the HBM roof that Marlin already reaches. The gain lives only in the AR-window chain arm and is capped by KB78/KB79's coupling.

## Other costing done (CPU only, no new measurement)
- PP=2 with deeper or tree speculation: the drafter cost scales with depth, so it is worse than TP. This agrees with T18.
- INT8/INT4/FP16-accumulate exact slicing: dead. The products' exponent spread forces too many slices. This agrees with T10 and T18.
- Algebraic AR removal (deferred norm, RS+AG, replicating out-projections): none of them reduces bytes or AR count.
- Mixed-step tensor-floor gap: by my count the floor is 24.36 GFLOP/token/rank = 0.167 ms/token on GPU0, against 0.229 measured. That is ~6% of wall above the tensor roof, spread across the T19 items. It is exploit-sized.

## Opinion
I have no other direction whose payoff if it works clears +5% with measured parts. I return ~19.5 GPU-min.
