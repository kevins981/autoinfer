# T17 write-up (leader-17, explore). Voluntary close after the seed

GPU: ~0.7 of 20 used (worker leader-17.w1, jobs 126, 127, 129-131, 133). Record: bank/T17-001. Notes: local_kb/leader-17.md.

## Direction (proposals/T17-1.md, approved in steps): a roofline decode GEMM for M<=64, cashed through k=3
- Premise: T6-001 put Marlin at 1.22x (M=48) and 1.39x (M=64) of the HBM roof. M=64 sits at the ridge, so a kernel that overlaps
  stream and math would make M=64 cost what M=48 costs. That removes the 3->4 m-block cliff that killed k=3 (T5).
- Forecast: payoff if it works +6.3%, central +1.5%.

## Result (T17-001): M0 killed, and the premise was a measurement artifact
- **Flush method.** T5-002/T6-001 measured "cold" with a 256 MB *write* flush. Its dirty lines are written back during the GEMM.
  - Marlin gate_up M=48/64 with a write flush: 110.6/124.1 us (reproduced in our job).
  - With a *read* flush: 92.5/103.5 us.
  - With the read flush, Marlin at M=48 is at the HBM roof (0.96 TB/s), and M=64 is only 1.11x the roof.
- **New kernel** (stream-K, 16 warps, 256-col tile, 4 m-tiles per dequantized fragment; exact to fp32 reorder, deterministic):
  - math alone at M=64 runs 88.6 us = 130 TF (89% of peak);
  - the B stream alone runs at only 0.84 TB/s (T8 got 0.75);
  - the stream-K fixup tail is 6-7 us;
  - full kernel 125.8 us vs Marlin 103.5.
- **Ceiling.** At M=64 the math (~89 us) and the HBM time (93 us) are nearly equal. A perfect kernel would therefore land at
  ~97-100 us, which recovers at most 1/3-1/2 of the 48->64 step. k=3's payoff ceiling falls to ~+2-3%, below the bar.

## Measurements worth keeping
- KB candidate: compare "cold" GEMM numbers against roofs only with a read flush. A write flush adds ~18-20 us per 89 MB GEMM.
  Earlier headroom claims built on T5-002/T6-001 cold sums are overstated by roughly that much.
  This does not apply to in-situ numbers.
- Note: in situ the 48->64 cliff is +3.0-3.4 ms/forward (T5-006), larger than the read-flush microbench step (~+11 us on
  gate_up). A plausible cause is dirty GDN state writes landing in the next GEMMs (KB14/KB15 spill). This is unmeasured.
- Reusable: bank/T17-001/code/g17.cu, an exact stream-K reader of Marlin's FP8 layout, with an ablation harness
  (math-only / B-only / A-only, in-kernel globaltimer per CTA).

## Opinion
- With a clean flush, decode Marlin is at the HBM roof at M=48. At M=64 it is within ~11% of a roof that the math alone nearly
  reaches. No GEMM-side route makes extra verify rows cheap on this card.
- I agree with T2-T18: the remaining gains are 1-2% exact items.
