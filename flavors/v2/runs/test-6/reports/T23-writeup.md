# T23 write-up (leader-23, explore). Closing after the seed: package fell below the bar

GPU: ~0.6 of 20 used (one microbench job, leader-23.w1). Records: bank/T23-001..T23-003. Worker: leader-23.w1 (reports in agents/leader-23.w1/W1-1..W1-3/). Notes: local_kb/leader-23.md.

## Direction (proposals/T23-1.md, approved, start grant in tranches gated on G2a)
A stacked package of exact per-step-overhead items on T14-004: M1 drafter to its byte roof, M2 small mixed steps (Marlin slab padding + glue), M3 slot lifecycle, M4 GPU0/GPU1 rebalance. Payoff if it works +5.9%, central +1.9%.
Before proposing I re-costed PP=2 (+deeper k), two-microbatch decode overlap, exact INT8/FP8 slicing, replicated drafter/mixers, roofline GEMM + k=3 and more prefetch from first principles; I agree with T2-T22 that each is dead or capped (<3%).

## What was measured (all on T14-004)
- **Drafter is at its floor (T23-001, CPU, B1 nsys).** After the last target layer a decode step spends 2.74 ms: target lm_head 1.01, draft lm_head 2x306 us (1.16x byte roof), MTP Marlin 0.41 (~roof), fc 0.15, drafter host collectives 0.25. All collective waiting is on rank 1; rank 0 never waits. The skew comes from GPU0's lower clock on ~35 tiny sampling/rejection kernels (1.7 vs 1.4 us each), not an imbalance. M1 <= ~0.3%.
- **The "3.1 ms decode gap" (T19 L2) is an artifact:** one 1.5 s decode->mixed idle gap averaged over 495 steps. The real inter-step gap is 0.74 us median; decode period = span.
- **Small mixed steps (pt<256; n=27, M~132):** 67 ms vs 23.5 decode. Excess: Marlin +20.5 ms, comm +16 (matches the transport model), 587 vs 46 glue launches/step (+2.4 ms, ~11 per GDN layer: elementwise/gather/cat), chunked-GDN prefill +2.3, in-span idle +3.1, act/norm +0.9, attention +0.6.
- **Marlin slab padding (T23-002, G2a FAILED by estimate):** dead m-blocks average 0.74 per small step (0.97 over all mixed steps) at <= ~22 us/block/layer -> <= 1.0 ms per small step vs the 1.5 ms gate.
- **Down/qkv remainder launch (T23-003, read-flush microbench, FAILED):** T6's stock choice for down/qkv at M>128 holds under a READ flush; slab is never meaningfully faster; best cherry-pick <= 37 ms/trial.
- **Marlin sees cudagraph-padded M** (steps of 32) in mixed steps, not pt+dt; at padded N>=256 the token split runs each GEMM twice per layer (8 launches/layer). Analyses that use pt+dt as M understate computed rows.

## Forecast vs outcome
| item | forecast payoff | outcome |
|---|---|---|
| M1 drafter | +1.2% | <= 0.3% (measured near floor) |
| M2a slab padding | +0.9% | <= ~0.5% (estimate, gate failed); remainder variant ~0 |
| M2b glue, M3, M4 | +3.8% | not reached |
Package payoff if it works fell to ~+4.1% < bar (1004.4) -> closure condition agreed with the manager.

## Left for an exploit owner (each ~1%, none measured in situ by T23)
- Small mixed-step glue + in-span idle: ~5 ms per small step on ~220 steps/trial (~1%); the 587 glue launches are listed by kernel in agents/leader-23.w1/W1-1/q2*.md.
- Capture-grid padding: mixed steps compute up to 31 padded rows (~0.18 ms/row compute-bound); a finer grid was tried (KB44/45: every-16 grid OOMs above 512, null in the T1-033 stack).
- M3 lifecycle (<=1.1% dead slot + drain k), M4 rebalance (KB8: ~0 once decode is charged).

## Opinion
The champion's per-step overheads I could size are at or near floors; what remains is a scatter of <=1% exact items. I agree with T14/T18/T19: they are exploit-sized, best harvested one at a time with in-situ nsys A/B then ABBA.
