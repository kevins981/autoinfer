# T10 write-up (leader-10, explore). Closed voluntarily after the seed

GPU: ~0.0 of 20 used (jobs 85, 87). Record: bank/T10-001. KB: KB69. Notes: local_kb/leader-10.md. No worker started.

## Measured kill: exact FP8 tensor-core GEMMs (T10-001)
- Idea: split each bf16 activation exactly into hi/lo FP8 pieces (Ozaki-style) and run e4m3 x e4m3 mma.sync with f32 accumulation. That is 2 passes at 2x the bf16 rate with no weight dequant, so it would lift the M=48 decode GEMM's tensor ceiling (Marlin 68% of HMMA peak, KB66).
- Rate (TFLOP/s, GPU1 / GPU0):

| MMA | GPU1 | GPU0 |
|---|---|---|
| bf16 | 164.5 | 145.7 |
| e4m3, f32 acc | 329.0 | 291.3 |
| e4m3, f16 acc | 653 | 578 |

- Precision: max |err| / sum|p| on random e4m3 data (bitwise-equal outputs: 0/128):

| K | e4m3 MMA | bf16 MMA, same values |
|---|---|---|
| 128 | 3.0e-4 | 2.9e-7 |
| 1280 | 5.0e-4 | 9.5e-7 |
| 5120 | 3.5e-4 | 1.6e-6 |

- One MMA, C=2^p plus a product of 1.0: exact up to p=12, the 1 is lost from p=14. C=1 plus a product of 2^p behaves the same.
- Interpretation: the ".f32" FP8 accumulator aligns every addend, C included, to a ~13-bit window below the largest one (as reported for Hopper). So it is not fp32 accumulation. An exact split is impossible, even with promotion after every MMA. The exact tensor roof stays bf16: 146 TF on GPU0 at its locked clock, 165 TF on GPU1. INT8 needs >= 4-6 passes for exactness (T2); f16 accumulation is lossy.

## The coupling map: why the three levers share one cap
- Decode step (M=48), per GPU:
  - HBM: linears 12.25 GB + target lm_head ~1.0 GB (encoded) + GDN fp32 states ~1.2 GB + drafter ~0.9 GB = ~15.3 GB, ~16 ms at 0.95 TB/s.
  - Tensor: 1.2 TFLOP. At Marlin's 68% of 146 TF that is ~12 ms, about the same as the GEMMs' HBM time.
- So the M=48 GEMMs sit on both roofs at once. Warm (L2-resident) time is only ~8% below cold time (KB66). That one fact caps all three remaining levers:
  1. **Hiding ARs** (KB62): the AR window can only be filled with L2 prefetch for the next GEMM. A prefetched GEMM is tensor-bound, so it saves only the ~8% warm-vs-cold gap (the ~18 us/AR the champion already gets).
  2. **Deeper speculation** (KB63): extra rows above M=48 cost tensor time at once (+3.4 ms for M=64).
  3. **Lossless weight compression**: fewer bytes don't help a GEMM whose tensor time already matches its byte time. Cap ~1 ms/step even with perfect compression.
- Raising small-M bf16 tensor efficiency would unlock all three, and that is what T8 failed to do (KB68).
- Prefill: ~104.5K prompt tokens x 24.5 GFLOP/GPU at ~125 TF is ~20 s of the 96 s trial. The measured increment is ~0.27 ms/token vs ~0.196 at the floor, so ~7.6 s excess (prefill AR exposure, T9; small-step overheads, T7).
- Slot accounting (replay of T1-042 official trial 1, reproduces its 96.3 s): drain 3.8 s = 2.3% slot loss. Mean TTFT 220 ms (median 167) = 5.7% of capacity, of which <= ~3% is above a ~105 ms floor.

## Costed items below the bar (estimates, none measured end to end)
| item | size |
|---|---|
| Copy-engine decode AR, pipelined, beside a separate L2-prefetch kernel (T8's leftover; DMA coexists with HBM traffic, SM host traffic doesn't) | ~1-3% |
| TTFT/admission lag (avoid the one-step async scheduling lag; merge small prefills) | <= ~2-3% (part of the 5.7% dead time) |
| Exact top-1 target lm_head: bit-plane storage (no extra memory), screen with the top planes plus a rigorous error bound, exact recompute of the candidates with the champion kernel's arithmetic | ~30% of lm_head bytes, ~0.3 ms/step, ~1% |
| Better AR payload compression (0.781 -> ~0.70 entropy bound) | ~1% |
| Lossless FP8 weight code inside Marlin (entropy 0.82, fixed-rate practical ~0.9) | <= ~1 ms/step before decode ALU cost, likely ~0 |
| Adaptive deeper k when few requests are running (drain) | ~0.8% |
| Draft lm_head vocab 98K -> 32K | ~0.5% net |

## The only >=5% construction, and why I didn't propose it
- Three pieces:
  1. A custom small-M bf16 GEMM: swap-AB (tokens on the n8 dimension), 2 warps/scheduler, ~85-90% of peak, streaming at the HBM roof.
  2. Compressed FP8 weights decoded inside it.
  3. Two-chunk decode ping-pong: two request chunks share each weight read through L2, so one chunk's AR runs on the copy engines while the other computes.
- Payoff if all three work: ARs mostly hidden plus ~1-2 ms fewer GEMM bytes, about +8%.
- Odds well under 5%:
  - Piece 1 alone is T8's failed kernel (gate_up 129.5 vs Marlin 87.8 us).
  - Piece 3 needs L2 residency across consecutive ops that don't fit L2 (gate_up 89 MB, down 45 MB).
  - Piece 3 also needs no tensor padding: Marlin pads 24+24 rows to 64.
  - It runs into the zero-sum interference already measured (KB59, KB62).

## Opinion
- The champion's decode step is pinned by coupled HBM and bf16-tensor floors, and FP8 tensor cores are not an exact way around them.
- The remaining gains are 0.5-3% items, better suited to the exploit team. T9's prefill AR hiding is the largest open one.
