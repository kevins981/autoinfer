# T8 write-up (leader-8, explore). Recommending a voluntary close after M1

GPU: ~2.0 of 80 granted used (20 seed + 60 grant); seed jobs 61, 62 (~0.1); worker jobs 63-77 (~1.9).
Records: bank/T8-001..T8-003. Worker: leader-8.w1. KB: KB66 (manager).

## Direction tried: a 2-warps/scheduler FP8 decode GEMM (proposals/T8-1.md). Killed at M1
- Premise, measured (T8-002, job 62):
  - A Marlin-like inner loop at M=48 runs at 68% of the HMMA peak with 1 warp per scheduler (= Marlin's warm rate).
  - With 2 warps per scheduler the same loop reaches 91.5%.
  - The L2 cp.async feed reaches 4.2 TB/s.
- Full kernel (T8-003, worker):
  - Exact (fp32 order only), deterministic.
  - Warm gate_up 129.5 us vs Marlin 87.8 (gate <= 72); cold 131.7 vs 95.0.
  - The loop alone keeps only 79% inside a realistic CTA. The weight stream sustains only 0.75 TB/s.
  - Marlin cold (95 us for 89 MB) sits at the HBM roof.
- Forecast vs outcome:

| part | forecast | measured |
|---|---|---|
| warm loop | 85-91% of peak | 79% loop-only, ~57% with the stream |
| stage 1, per decode step | -2.0 ms | positive (slower); ceiling at 79% <= -1.3 ms (~+2.8%) |

- My error:
  - I took a toy loop (12 independent MMAs per warp, no global stream, no barrier) as the reachable rate.
  - T1-022's 8-warp Marlin config gaining only 2-3.6 us was the warning. I discounted it.

## Other results (all measurement only)
- **T8-001 (job 61). Pinned host page size doesn't matter.**
  - DMA copies beside an HBM-saturating kernel slow only x1.01-1.12, and the kernel x1.00-1.04.
  - SM-driven host traffic (hostar style) slows x1.45 and costs the kernel 6-19%.
  - The decode hostar runs beside its own L2-prefetch traffic, which is the bad case. A copy-engine decode AR might hold
    speed there (untested in-server).
- **Trace (agents/leader-8/tl3.py on T2-006).** Token-split mixed steps (R>=256) spend 15-23% of their span with only
  comm running: both chunks' ARs fire in lockstep. That's ~3% of the trial (T9-1 now targets it).
- **CPU-only kills:**
  - prompt-lookup drafting: oracle at k=2 gives E +0.9% on T3's self-generated data (agents/leader-8/ng/sim.py);
  - CPU AMX prefill offload: oneDNN bf16 reaches ~7 TF on 12 cores, ~5% of the GPU pair (agents/leader-8/cpu/mm.py).

## Reusable
- agents/leader-8.w1/g2w.cu: a correct reader of Marlin's FP8 repacked layout and scales, with bit-identical dequant.
  The layout notes are in the worker's report (T8-003 summary).
- agents/leader-8/pg/pg.cu, tc/tc.cu: PCIe and tensor-pipe microbenches.

## Opinion
- The M=48 decode GEMM is at Marlin's roof cold and close to a realistic 2-warp ceiling warm. A new GEMM is worth at
  most ~1 ms/step, so it's not an explore direction.
- The remaining untested angle from this team: a copy-engine decode AR beside the L2 prefetch (T8-001). It's exploit-sized
  (<= ~1-2%).
