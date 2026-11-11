# leader-2.w16 round 9: standalone W8A16 decode GEMM prototype (Marlin layout) vs stock Marlin

Files: k/w8dec.cu (v1), k/w8dec2.cu (v2: 32-bit indexing, debug `mode` knob), k/build.py (CPU-side JIT, build dir by source hash),
k/bench.py + k/job2.sh (bench: same method as leader-2.w10/mb: GPU 0, CUDA graphs; cold = cycle >= 320 MB of copies;
warm = prefetch.global.L2::evict_last of the server MB right before + globaltimer stamps, 0.77 us stamp overhead subtracted).
Results: k/r1 (job 106), k/r2, k/r3, k/r4 (log.txt, res.json).

## Kernel design (w8dec2.cu)
- Swap-AB: weights are the MMA A (m16) operand, tokens the n8 side (M=48 -> 6 n8 tiles), mma.m16n8k16 bf16, fp32 acc.
- Reads Marlin's repacked qweight and Marlin-permuted bias-folded bf16 scales as-is (no repack): a lane's 32 B in a
  16x64 Marlin tile = 8 n8-tiles x [k0,k8,k1,k9]; two adjacent n8 tiles form one m16 A fragment.
- Dequant bit-identical to Marlin (shift trick + __hmul2 by bf16 scale -> HFMA2.BF16 with -0).
- 8 warps/CTA (2 per scheduler): cfg0 = 4 warps along N (256 cols) x 2 along K, KE=64/stage, 4 stages (90 KB smem);
  other cfgs: 1 = 8x1 (512 cols, KE32), 2 = 4x2 KE32 6 stages, 3 = 2x4 (128 cols), 4 = 4x1 128 thr (2 CTA/SM).
- Persistent stripes over (n-block, k-unit) like Marlin; each segment writes per-warp fp32 partials; deterministic
  cross-CTA reduction (fixed order) once all partials of an n-block arrived.

## Correctness (r1, r2): OK
max |mine - Marlin| = 1 bf16 ulp (0.0156-0.031 at |y|~6.6; 0.0-0.1% of elements differ), max |mine - fp32 ref| equals
Marlin's (1.56e-2); bitwise deterministic run to run. All shapes/M tested.

## Timing (us, GPU 0, M=48 unless noted; Marlin = champion build default config, same job)
Debug modes in w8dec2.cu: mode 5 = full main loop (loads + dequant + MMA) but no epilogue/reduction; mode 7 = load
pipeline only (cp.async + barriers, no MMA); mode>>4 = L2 prefetch distance (stages).
| r1/r2/r5 (cfg0)          | gate_up cold | gate_up warm | in_proj cold | in_proj warm |
| Marlin                   | 107.5-112    | 83.9-84.5    | 51.3         | 42.8-43.8    |
| mine full (mode 0)       | 155.6        | 129.2        | 65.9         | 59.3-60.5    |
| mine no epilogue (5)     | 111.9        | 98.7-100     | 56.1         | 49.6-51.0    |
| mine loads only (7)      | 102.6        | 88.6-89.0    | 47.9         | 42.0-42.7    |
M=16 Marlin warm: gate_up 72.8, in_proj 31.7; mine loads-only 86.4 / 40.7-41.2 (no compute!). M=32 (r1 v1) similar to M=48.
- All configs (r1, r4: 256/512/128-col tiles, KE 32/64, 4-6 stages, 128-thr 2 CTA/SM, grid 128/256) land within a few %:
  loads-only warm gate_up 86-93, in_proj 41-45. Cold loads-only is 4-7% faster than Marlin (near the HBM roof).
- L2 prefetch-ahead (r3, PD 4/8/16/32 stages): cold -3.5 us at PD4, warm +8..+12 us worse.
- Warm with a 45 or 120 us memory-idle spin between prefetch and GEMM (r5): identical numbers for both kernels, so the
  prefetch had landed; the warm gap is not prefetch-still-in-flight.

## Interpretation
1. Epilogue: per-warp fp32 partials = ~25 MB for gate_up (128 CTAs x 2 segments x 96 KB): +30 us warm, +44 us cold (cold:
   the mid-kernel partials get evicted to HBM by the weight stream). Fixable (in-CTA K reduce, 128-col tiles -> ~6 MB,
   evict_last stores), but that only gets back to the "mode 5" line.
2. Compute is NOT hidden by 2 warps/scheduler: mode 5 - mode 7 = +11 us gate_up, +8 us in_proj warm.
3. Decisive: even with no MMA at all my load pipeline is slower warm than Marlin's whole kernel (and 20-30% slower than
   Marlin's M=16 run), across every tile shape tried. "Warm" here is a partial-L2-hit regime (pure streaming read of
   prefetched in_proj = 26.9 us = 1.56 TB/s, far below L2 bandwidth), where throughput depends on request concurrency /
   order in ways I could not pin down in 5 jobs. Reaching the gate (gate_up <= ~72, in_proj <= ~36.5 warm) would need a
   data path better than Marlin's M=16 run plus fully hidden MMA plus a near-free epilogue: clearly out of reach this round.
GPU used: ~5 min (jobs 106 + r2..r5).
