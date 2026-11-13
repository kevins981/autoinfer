# LOCAL_KB leader-53 (T53, exploit-long, megakernels; human-set direction)

## Cards (first set)
Boundaries + "every transfer has a fixed cost" + "more than one kernel at a time": the AR is a boundary where the whole
64x5120 row must cross before anything downstream starts; moving the boundary to chunk granularity lets transport overlap
the producing/consuming GEMMs. Predictability/repetition: nothing obvious for decode megakernels.

## Anatomy (0 GPU, agents/leader-53/anat/, T51 jobB nsys of T45-005, GPU0, 89 decode steps at 64 rows)
- step 25.06 ms: target fwd 22.1, lm_head 1.03, verify ~0.13, drafter 3 levels ~1.75 (0.67/0.54/0.54).
- GDN layer ~360-372 us: hostar 52 + norm/quant 5 + qkvz t31b 49 (ba bf16 GEMM beside) + conv 4 + GDN replay 40 +
  out t31b 22 + hostar 56 + norm/quant 5 + gate_up bs_gemm 81 + silu-quant 3 + down t31b 53. Attn layer ~324 us.
- hostar 141 calls x 49.8 us = 7.0 ms/step. DRAM floor GDN layer ~230-250 us (192 MB weights + 25 MB state).
- AR is PCIe-bandwidth-bound: ~1 MB host traffic per AR at ~22 GB/s ~ 46 us.

## Sparring T53-1 (beliefs)
- bf16 r-deferral: allowed. FP8 r-deferral: manager RULED ALLOWED 2026-11-11 (report diffs, FP8 GSM8K A/B, 128K).
- chunked producer/consumer keep throughput: to test (<=10% gate).
- coupled overlap: to test; kill if coupled chain saves < 40 us per AR boundary (full pipeline) / producer-only gate 15 us.

## Coordination
- T52 (leader-52) = CONSUMER side (AR read + add/norm/quant + next GEMM, weights streamed in window).
- T53 = PRODUCER side (chunk-ordered row-parallel GEMM posting chunks during GEMM) + chunk-flag interface
  (agents/leader-53/INTERFACE.md, to write).
- T52 fact: AR stretches only ~+2 us with 16 real-load prefetch blocks streaming 32 MB (T31-001).

## Worker
- leader-53.w1 (ws t53-prod = T50-002): task agents/leader-53/w1_task1.md, allowance 8 GPU-min.
- T52-001 (chain microbench M=64): AR in a resident 128-CTA persistent grid 44-53 us vs hostar alone 50.4; streaming
  42 MB weights during AR +10 us at 16 AR blocks (+21 at 8); stream capped 11-19 MB free; whole-row consumer fusion
  LOST (103.5 vs ~89 us; hint prefetch already gets qkvz to warm floor). T52 next: K-chunk cut-through consumer.
- INTERFACE.md v0 written (agents/leader-53/INTERFACE.md), sent to T52.
## M1 results (worker, jobs 441/442/445, 1.1 GPU-min; agents/leader-53.w1/RESULT.md)
- Gate A PASS for down: chunk-ordered persistent t31c C2 1.054x, C4 1.081x (chunks at 18/33/47/58 us of 62); C8 1.36x.
  out_proj only C2 (1.085x). Needs next-item first-block preload (v1 without it 1.14x+). kseg=1 bitwise.
- Gate B FAIL: SM posted host writes slow the HBM-bound GEMM +7-9 us even from dedicated poster CTAs (+25 from the
  tile-finishing CTAs, = KB72); posters only ~16 GB/s -> 10-16 us tail. Napkin C: down saves ~5-12 us/boundary < 15.
- Round 2 asked: hostar write pattern + codec, ablation (no host stores / device stores) to locate the slowdown,
  CE path via wait-value on device counter, then coupled benchC. +3 GPU-min.
- My napkin if (a)+(c) work: down chain ~81 vs ~100 us -> ~19 us/boundary x 64 = ~1.2 ms/step ~ +3.5% e2e alone;
  needs T52's consumer cut-through to clear +5%.
## Round 2 (jobs 446 hung 4 min - wait-value enqueued before producer; 448 ok). Total T53 used ~5.2 GPU-min.
- Host-store slowdown of the GEMM is purely PCIe host stores (no-store / device-store ablations ~0), grows with bytes+rate.
- codec x8 posters: C4 +3.1 us / tail 9.2; C2 +4.1 / 14.3 -> Gate B pass. 16 posters worse. Wire ~18 GB/s during GEMM.
- Copy engine via wait-value: no GEMM slowdown but lands 17-20 us after flag (eager or graph) -> out.
- benchC projection ~15-17 us/boundary (gate 15). Running benchC incl. gate_up after AR (prefetch fairness), 1.0 min.
- benchC job 450 died (lazy module loading + spinning barrier, 1.0 min lost); job 452: champion chain down->AR 108.5/110.6
  us, +qkvz 143.4/146.4 (qkvz 35 with prefetch, 50 without; AR after down prefetches qkvz+out_proj, not gate_up).
  Fused C4/C2 ~250 us: GEMM 3x slower on 2 GPUs (suspect 8 reader CTAs spin-polling host flags, KB105).
  Rerun approved 0.3 min: single poller + no-poll-until-own-done variant; stop if GEMM >1.3x single-GPU.
## Gate C (job 454; T53-003): uniform C2 codec saves 9.7-12.3 us/down boundary (down->AR 113 vs 123; chain to qkvz
148 vs 159-161); C4 3-7 us. GEMM on 2 GPUs 1.09-1.31x single-GPU; incoming host READS slow it 6-14 us; prefetch kept.
T53 used 7.0 GPU-min (balance 13.0) before tapered round.
- My reading: the limiter is the TAIL (~41 us after GEMM-done at C2 = post last 256 KB chunk + peer read), which
  scales with the LAST chunk's size -> tapered chunks [big..., 640]. Predicted down->AR ~90-95 (save ~28-33).
  Asked T53-004 (1.2 min). If it holds: per-boundary ~30 us x 64 down + consumer-side (T52) on out_proj ->
  payoff-if-works ~+7-8% e2e.
- T53-004 (job 458): tapered chunks all WORSE than even C2 (C2 +8.7/+9.2 this job). My prediction missed: split-K
  chunks slow under host traffic; effective link ~13-15 GB/s during GEMM vs ~23 alone; small last chunk waits behind
  the backlog. Cause = SM-driven link throughput during GEMM, not last-chunk size.
- Last producer variant: hybrid (chunk0 via copy engine, which costs GEMM ~0 and whose ~20 us latency is hidden by
  the remaining GEMM; chunk1 via SM posters). Pred down->AR 100-105, gate >=18 us. T53-005. Fail -> stop producer side.
- Lean-megakernel tally for proposal (assumed unless noted): producer posting 0.64 ms/step (MEASURED ~10 us x 64),
  GEMM boundary prologue/tail overlap 0.5-0.8 ms, small-op fusion ~0.5 ms, drafter overhead ~0.5 ms.
- T52-002: decode consumer cut-through at PARITY (89-91 vs 88-92 us); chunk usable ~20 us after post. T52 pivots to
  mixed/prefill steps; T53 keeps decode (agreed). => full decode AR pipeline (spar gate 40 us) effectively dead.
- T52 (proposals/T52-1.md): champion-era node trace shows token-split halves' GEMMs run concurrently, so both CE ARs
  (~260-280 us each) are exposed: ~550 us/layer, ~40% of a 380-row mixed step. KB67's 86-96% was microbench only.
- T53-005 (job 459): hybrid CE chunk0 + SM chunk1 FAILS (saving +2.6-4.1 = same as C2 this job; baseline varies
  108-124 us across jobs). Last-chunk post+read tail unchanged. PRODUCER SIDE CLOSED: best 3-12 us/down boundary.
  Cause measured: SM-driven host traffic during the HBM-bound GEMM runs at ~13-15 GB/s and slows the GEMM; CE path has
  ~20-32 us latency. (direction-limit for decode AR overlap with today's link mechanisms; T52's consumer side = parity.)
- Next: T53-006 ceiling test in T51 chain harness: small-op ceiling, boundary ceiling (merged GEMM stream), GEMM
  inefficiency vs DRAM floor. Decides whether a lean decode-layer megakernel proposal can clear +5%.
- T53-006 (job 462, ceil harness): per layer i 387.4 (V0b champion-like 342.1), small ops removed -19.5, +AR removed 277.8,
  merged 2-GEMM stream WORSE +20.8 (boundaries ~0 in graphs), DRAM floor 234.5; GEMM-vs-stream gap ~33 us/layer.
- Sparring T53-1 end: closure is defensible, but half-batch interleave UNTESTED (my additive bound ignored prefetch in
  A's own windows). -> T53-007 optimistic bound (spins, 2 streams, M=32 halves) vs V0b, gate <=0.92x.
- T53-007 (job 464): half-batch 1.31-1.47x V0b even with spin ARs (M=32 GEMMs ~= M=64 cost, little L2 reuse). Closed.
- DECISION: recommended closing T53 to manager (no proposal). Write-up agents/leader-53/WRITEUP.md. 9.2/20 GPU used.
