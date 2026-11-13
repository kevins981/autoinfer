# T53 write-up (exploit-long, leader-53; direction set by the human: megakernels). Recommendation: CLOSE.
Records: bank/T53-001..007 (all microbench, measurement only). Worker leader-53.w1 (agents/leader-53.w1/RESULT.md).
GPU: 9.2 of 20 seed (5 of it lost to two hung/failed jobs: 446 wait-value order, 450 lazy module loading). No proposal.

## Starting point (0 GPU, agents/leader-53/anat/, T51's nsys of T45-005, 89 decode steps at 64 rows)
Step 25.1 ms: target forward 22.1, lm_head 1.0, drafter 1.75. GDN layer ~360-372 us vs ~235 us DRAM floor; the gap is
~2 host ARs (141 x ~50 us = 7.0 ms/step) + ~15-20 us small ops. The AR is PCIe-bandwidth-bound (~1 MB/AR at ~22 GB/s).

## Direction tried: pipeline the decode AR at chunk granularity through the GEMMs (T53 = producer side; T52 = consumer)
- Chunk-ordered row-parallel GEMM (t31c: t31b inner loop, persistent CTAs walking chunk-major items, last-arriver
  split-K reduce, next-item block preload) keeps throughput on down: C2 1.054x, C4 1.081x; out_proj only C2. (T53-001)
- Posting chunks to pinned host DURING the GEMM: SM host stores slow the HBM-bound GEMM in proportion to bytes and
  rate (no-store / device-store ablations cost ~0); codec + 8 posters +3-4 us; copy-engine D2H via wait-value on a
  device flag costs the GEMM ~0 but lands 17-32 us after the flag. (T53-002, T53-005)
- Coupled 2-GPU chain down->AR->qkvz vs champion hostar (codec + prefetch), same job: best saving 3-12 us per down
  boundary across 4 jobs (gate 15-25). Tapered chunks worse; CE first chunk no better. While the GEMM runs, SM-driven
  link traffic reaches only ~13-15 GB/s (vs ~23 alone) and incoming reads slow the GEMM too; the last chunk's
  post + read tail always remains. Ceiling ~0.2-0.7 ms/step (<=~2% e2e). (T53-003..005)
- T52's consumer cut-through: parity (T52-002). The full pipeline (sparring gate >=40 us/boundary) is out of reach.
- Baseline drift: champion down->AR varies 108-124 us between jobs: compare within a job only.

## Ceilings for the rest of a decode megakernel (T53-006/007, T51 chain harness, ARs as spins)
- Per layer: chain 387 (champion-like with in-AR prefetch 342); small ops removed -19.5 us (gross, spins without
  memory traffic); boundaries ~0 in graphs (merging 4 GEMMs into 2 launches is 21 us/layer WORSE); DRAM floor 234.5;
  GEMM-to-stream gap ~33 us/layer is GEMM-kernel efficiency, not a boundary effect.
- Two offset half-batches (M=32 x2): 1.31-1.47x even with interference-free AR spins: M=32 GEMMs cost ~M=64 GEMMs and
  B gets little L2 reuse (16-89 MB weights vs 72 MB L2). Closed.
- So megakernel-specific reachable items at decode sum to <=~1.5-2 ms/step ideal (~5% e2e at best, realistic well
  under), below the +5% payoff bar.

## Kind of failure
Direction hit measured limits on this hardware for the decode pass: (a) PCIe host traffic and HBM streaming interfere
(SM path) or carry 20-30 us latency (copy engine), so the AR can't be hidden behind its neighbouring GEMMs; (b) launch
boundaries already cost ~0 under CUDA graphs. Not proven impossible for every schedule (sparring: a universal ceiling is
unestablished), but every tested schedule is under the bar, and the remaining untested one I know of (tile-locked
half-batches holding weights in L2 with evict_last across the AR lag) has the L2-capacity and M=32-cost problems shown
in T53-007.

## For later teams
- Reusable: t31c chunk-ordered GEMM (agents/leader-53.w1/ext/t53c.cu), codec posters, single-poller host flags with
  __nanosleep + device-word relay (many-CTA host polling made the GEMM ~3x slower), 2-GPU coupled bench benchC.py
  (warm every kernel on both GPUs before the first host-flag barrier: lazy loading blocks the host otherwise),
  ceil harness bench_ceil.py / bench_half.py. Interface draft agents/leader-53/INTERFACE.md (used by T52).
- Mixed steps (longer GEMMs) are where chunked transport may still pay: T52 holds that direction.
