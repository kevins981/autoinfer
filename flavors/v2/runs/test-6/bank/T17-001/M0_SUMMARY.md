# T17 M0 summary (leader-17.w1) — KILL (kernel built, correct, exact; load path 0.84 TB/s, gate arithmetic out of reach)
GPU0, workspace t17-k (built from T14-004; vllm resolves to workspaces/t17-k), gate_up 17408x5120, random FP8 weights.
Jobs: 126 (setup error), 127 (bench b17.py), 129-131 (ablations ab17/ab2/tm17), 133 (load-path ablation). ~2 of 5 GPU-min used. Outputs in gpu/jobs/<id>/output.log and final/job*.out.
## Kernel (final/g17.cu)
Stream-K, 128 CTAs x 16 warps, tile 256 cols (4 Marlin chunks, 4 KB contiguous per k16), warp = 16 cols x all m16 tiles (4 m tiles per dequantized B frag),
4-stage cp.async ring (64-k stages: 16 KB B + 8 KB A + 0.5 KB scales; 98 KB smem), ldmatrix A, dequant identical to T8/Marlin, fixed-order fp32 partial fixup via L2 workspace + release/acquire flags. 127 regs, no spills.
## Exactness (job 127) vs stock Marlin, M=1/16/32/48/64
max|d| = 4.9e-4 (|y|<=0.19) at every M; 0.12-0.16% elements differ; >1ulp only 0.006-0.011% (near-zero outputs); bitwise deterministic run-to-run; flags self-clear.
## Timings, gate_up (profiler kernel durations, 256MB READ flush; med of 17; job 127), us
M=64 cold: Marlin 103.5 / mine 134.4; warm(40MB prefetch) 108.9 / 134.3; chain(45us spin + concurrent 40MB evict_last prefetch) 102.7 / 133.9
M=48 cold: Marlin 92.5 / mine 116.6; warm 92.2 / 113.1; chain 86.0 / 112.8
## Marlin number reconciliation (job 130)
T5-method (8x[flush.zero_()+GEMM] graph minus flush-only; 256MB WRITE flush): Marlin cold M=48 110.6, M=64 124.1 (T5-002: 111.2/126.3, reproduced).
Read flush: 92.5 / 103.5. Difference ~18-20 us = write-back of the dirty flush lines competing for HBM during the GEMM. Both valid; read-flush M=48 is at the HBM roof (89MB/92.5us = 0.96 TB/s). Step 48->64: +11 (read) / +13.5 (write).
## Ablation, in-kernel globaltimer per CTA, M=64, read flush (jobs 131, 133) — median [max] CTA time in us
| variant | compute-done | end |
| full (NST=4) | 118.9 [127.4] | 125.8 [134.5] |
| full, NST=3 | 118.9 [125.0] | 125.4 [133.0] |
| no A loads | 111.7 [119.1] | 118.9 [126.2] |  (NST=3: 109.8 / 115.6)
| no B loads (A + math) | 98.7 [103.7] | 105.2 [111.3] |
| no loads at all (dequant+ldmatrix+MMA+barrier) | 88.6 [93.5] | 95.0 [101.2] |  = 130 TF
| loads only, no math (B+A) | 105.9 [110.5] | 112.6 [118.1] |
| B loads only, no math | 106.4 [110.3] | 112.5 [118.3] |
| A loads only, no math | 15.6 | 22.3 |
(end - compute-done = fixup: 6 us median, 15 max; includes waits on neighbours' partials. T5-method ablations M=64: no-fixup 130.3, loads-only 127.7, full 163.7 incl. ~20 us write-flush artefact.)
## Reading (Interpretation)
- Math alone is at the tensor roof (88.6 us = 130 TF vs 133 TF measured peak); HBM roof 93 us. Gate (M=64 <= Marlin M=48 +5% ~ 97 us) needs both roofs within ~4% plus all ramp/tail in <4 us.
- B stream alone reaches only ~0.84 TB/s (89 MB / 106 us) with 4 contiguous 4KB pieces per stage, 3 stages in flight per SM, regardless of math; NST 3 vs 4 changes nothing; A traffic is free. Same ceiling as T8 (0.75). Marlin gets 0.96 TB/s. I did not find the lever (not the barrier/ring depth/A/L2 hotspot); untested: no per-stage barrier at all, 2 CTAs/SM, different B access order across CTAs.
- Overlap works (math adds ~13 us on top of loads) and fixup is ~7 us; best realistic outcome with a perfect load path ~ 89 + 7 + ramp ~ 100 us vs Marlin 103.5 => recovers <= ~1/3 of the 48->64 step.
