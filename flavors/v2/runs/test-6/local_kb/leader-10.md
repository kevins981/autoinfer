# leader-10 (T10, explore) working notes

## Seed work, 2026-11-09 (jobs 85, 87; ~0.0 GPU-min of 20)
- bank/T10-001: e4m3 mma.sync runs at 2x bf16 (329 vs 164.5 TF on GPU1; 291 vs 146 on GPU0).
  Its ".f32" accumulation keeps only ~13 bits below the leading addend, C included (2^14+1 -> 2^14).
  So no FP8-MMA GEMM can be exact, even with an exact hi/lo split of the activations. Family killed.
- Order-0 entropy of the FP8 weights (layer 10, all linears): 6.53-6.61 bits/weight = 0.82 ratio.
  Exponent entropy is 2.6 bits; exponents 11-14 hold 85% of weights.
  A fixed-rate in-GEMM code reaches only ~0.9 (a 3-bit exponent class, ~1.5% escapes).
  At M=48 the decode GEMM's warm (tensor-bound) time is within ~8% of its cold time (KB66), so compression alone
  is capped at ~1 ms/step.
- Slot accounting from T1-042 official trial 1 (bench_c16.json latencies, closed-loop replay matches duration 96.3 s):
  - drain 3.8 s = 2.3% slot loss;
  - mean TTFT 220 ms (median 167) = 5.7% of slot capacity, of which ~115 ms/request is above a ~105 ms floor
    (half-step wait + the 91 ms mixed step), i.e. <= ~3%.

## First-principles map of the decode step (23.8-25.5 ms, M=48)
- HBM bytes/step/GPU: linears 12.25 GB + target lm_head ~1.0 GB (encoded) + GDN fp32 states ~1.2 GB
  (48 layers x 16 req x 24 heads x 64 KB) + drafter ~0.9 GB (2 x (MTP layer ~0.19 + draft lm_head 0.25)).
  Total ~15.3 GB, i.e. ~16 ms at 0.95 TB/s.
- Tensor: 1.2 TFLOP/step; at Marlin's 68% of the 146 TF GPU0 peak that is ~12 ms, about equal to the GEMMs' HBM time.
  So M=48 sits at both roofs, and the AR window can only be filled by L2 prefetch, which saves just warm-vs-cold
  (~8%).
- The coupled floors explain the ceilings earlier teams hit:
  - hiding ARs needs GEMM slack;
  - deeper speculation needs tensor slack;
  - compression needs tensor slack.
  All three are blocked by small-M bf16 tensor efficiency, and T8 failed to raise it.
- Prefill: ~104.5K prompt tokens x 24.5 GFLOP/GPU at ~125 TF on GPU0 = ~20 s of 96 s.
  The measured increment is ~0.27 ms/token vs 0.196 ms/token at the floor, i.e. ~7.6 s excess
  (AR exposure, which T9 targets, plus small-step overheads, T7).

## Candidates considered and why not >=5%
- FP8 split GEMM: inexact (T10-001).
- INT8 Ozaki: >=4-6 passes (T2).
- fp16-accumulate: lossy.
- 2:4 sparse split: no gain.
- Strassen/Winograd: bf16 sums aren't representable.
- Lossless FP8 weight compression: ~0.9 practical, capped by tensor coupling, ~1 ms/step.
- Exact top-1 target lm_head (bit-plane screening + exact recompute of candidates): ~30% of lm_head bytes,
  ~0.3 ms/step (~1%).
- Two-chunk decode ping-pong: needs a shared weight stream, and M=24+24 pads to 64 MMA rows in Marlin.
  Without a custom swap-AB GEMM it is worse.
- Adaptive deeper k at low batch (drain): ~0.8%.
- TTFT/admission lag: <= ~2-3%.
- Draft lm_head vocab 98K -> 32K: ~0.5% net.
- Better AR compression (0.78 -> ~0.70 entropy bound): ~1%.
- Copy-engine decode AR beside a separate L2-prefetch kernel (T8's leftover): ~1-3%.
