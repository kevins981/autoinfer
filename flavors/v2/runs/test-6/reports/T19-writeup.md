# T19 write-up (leader-19, explore). Recommending a voluntary close after the seed

GPU: 15.0 of 20 used. Job 132 lost 12.0 min to a tracing-hook bug (the engine died, the bench hung to timeout); job 134 took 3.0. Balance 5.0 is returned.
Records: bank/T19-001 (env-gated lifecycle/step tracer; measurement only). Worker: leader-19.w1. Notes: local_kb/leader-19.md.

## Direction examined: the new-request path
What each new request costs the system at c=16:
- slot refill latency (KB75);
- the mixed step the request creates;
- prefill coalescing.

## Measured (in situ, T14-004 + tracing, job 134, 400 prompts, 954.0 tok/s vs official 956.6)
Report: agents/leader-19.w1/L1/REPORT.md.
- **Decode steps:** 86% of steps, period median 25.3 ms.
- **Mixed steps:** 370 per trial, 40% of wall. Period = 38.5 + 0.229 ms x prefill tokens (R^2 0.98). Steps with <64 prefill tokens take a tight 49.7 ms.
- **Slot refill:** the replacement prefill is scheduled 3.23 steps after the finishing step (3 in 78% of cases, 4 in 22%), confirming KB75's N+3. But dead slot-time is only **1.1%** of 16 x trial, so the lifecycle item is <= ~1%. End-of-trial drain is 2.6%.
- **Request arrival path, means:**
  - render 11.6 ms;
  - input processing 3.6 ms;
  - engine queue 20.0 ms (median 5.4);
  - scheduled -> first token 156 ms;
  - client/HTTP 7.7 ms.
- **p99 TTFT** (server 1045 ms, client 1126 ms) is the ramp-up burst: the first 16 arrivals pile into 1734- and 2038-token steps. Steady-state p99 is ~0.41 s.
- **Coalescing** (Interpretation): a merge saves ~13 ms (one 38.5 ms intercept minus an added 25.3 ms decode step) and costs about as much in idle slot. Net ~0. Dead.
- **Co-landing:** 5.6% of admitting steps already carry 2+ new requests. No chunked-prefill continuations occur.

## Critical-path attribution of mixed steps (CPU only, T14's in-situ nsys B1 = T14-004, ranks 0 and 1)
Report: agents/leader-19.w1/L2/REPORT.md.
- **Decode, rank 0:**
  - Marlin 13.0 ms, exposed comm 5.6;
  - comm overlap with any compute is exactly 0;
  - nsys inflates the period ~5%.
- **Mixed-step excess over decode, per 400-prompt trial (rank 0, nsys-timed):** 33.7 s in total.
  - Marlin 22.0 s;
  - exposed comm 6.8 s;
  - GDN 1.6 s;
  - norm/act 1.4 s;
  - idle 1.5 s;
  - attention 0.3 s.
- **The mixed-step penalty is a Marlin penalty.**
  - Marlin goes from 13 to 28 ms at only ~60 prefill tokens: the compute roof for 108 rows is ~20 ms, so ~8 ms/step is excess at small R (T14's R<256 item).
  - Above that, Marlin adds ~0.2 ms per token, which is at the roof.
- **Mixed-step exposed comm.**
  - Part of it is rank skew: hostar spins for the slower-clocked GPU 0. At 400-800 tokens, rank 0 has Marlin 131 / comm 29 ms and rank 1 has 113 / 46 ms.
  - On the pacing rank 0, the excess over decode is ~6 s of nsys time, roughly ~5% of wall in real time:
    - ~2.4 s in <128-token steps (not hideable, KB65/KB67);
    - ~3.5 s at R>=256 (T9's area; more chunks with per-chunk cores might hide about half).

## Costed (none clears 5% alone; the package is the T14-1 shape)
| item | ceiling | status |
|---|---|---|
| slot-refill / late scheduling | <= ~1% | measured (L1) |
| prefill coalescing | ~0 | measured (L1) |
| R>=256 AR: more chunks + per-chunk cores | ~+1.5-2% | sized from L2; T9 estimated <= 1% |
| small-step Marlin excess (padding, in-situ) | ~+1% | L2 + T14 |
| GPU0/GPU1 rebalance in mixed steps | ~+2% (needs duplicated MLP slice) | T9, T18; L2 skew consistent |
| drain adaptive k | ~+0.6% | T10 |
| GDN/norm glue in mixed steps | <= ~1.5% | L2: 3.0 s total excess |
A package of all of these reaches ~+6% only if every item hits its ceiling. Honest central estimate ~+2%. I don't propose it as an explore direction.

## Measurements worth keeping (KB candidates)
1. **In-situ step model** (T14-004, 400 prompts):
   - decode 25.3 ms;
   - mixed = 38.5 + 0.229 ms/prefill-token;
   - mixed steps are 40% of wall;
   - dead slot-time 1.1%, drain 2.6%;
   - p99 TTFT is the t=0 ramp burst, steady-state p99 ~0.41 s.
2. **Mixed-step excess over decode is ~2/3 Marlin** (33.7 s = Marlin 22.0 + exposed comm 6.8 + GDN 1.6 + norm 1.4 + idle 1.5). Decode comm never overlaps compute. Mixed-step "exposed comm" on rank 1 is largely skew wait for GPU 0.
3. **Tooling:** bank/T19-001's T19_TRACE tracer gives per-step and per-request timestamps (core and frontend) at <~1% cost. `vllm bench serve` needs `--save-detailed` for per-request client TTFTs; lib.sh doesn't pass it.

## Opinion
- The new-request path is not a lever: refill is ~1%, coalescing ~0.
- What remains in mixed steps is Marlin at small R, AR at R<256 and rank skew, all sized above at ~1-2% each.
- These are exploit items. Their best owner is an exploit team, harvesting them one at a time with in-situ A/B the way T14 did.
