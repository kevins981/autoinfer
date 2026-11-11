# leader-11 (T11, explore) notes

## Direction: window lane (2026-11-09)
- Decode step rank 0 ~24 ms; host-AR waits ~5.9 ms/step (128 ARs x ~42 us), AR kernel = 8 AR + 8 prefetch CTAs -> ~110 SMs tensor-idle.
- Tensor-idle total per decode step (AR+GDN+attn+norm+lm_head) ~10 ms (42%).
- Prefill compute floor: 104.5K tok x 24.5 GFLOP/rank = 2.56 PFLOP ~17.5 s at 146 TF (GPU0).
- Needed lane rate for ALL prefill ~34 TF avg; for small prompts (<400 tok, 41% of tokens) ~13 TF avg.
- Window capacity estimate: ~80 TF in-window x 5.4/25.5 ~ 17 TF avg (if lane gets ~110 SMs, 75% fill).
- Small-prompt mixed-step increment ~12.5 s/trial (215x34 ms + 86x60 ms, from T7/T2-006).
- Payoff model: saved = absorbed increment - s x ~65 s decode wall; s = decode slowdown.
- Risks: L2-prefetch loss (~2.3 ms/step if lost totally), lane CTA spill into decode, lane prefill AR latency chain (128 AR/pass x ~100-200 us) -> need >=2 prefills or 2 chunks ping-pong, TTFT p99 for long prompts (keep >=400 in mixed steps), in-server interference x3 microbench (KB59).
- Engineering: decode graph with forked lane nodes; lane plan per step via graph node param updates or a device work-queue lane kernel; lane DMA ARs during decode GEMMs.

## Jobs
- T11-001 (worker leader-11.w1, ws t11-lane = T4-001 built): P1 microbench lane-in-window, 2-proc chain.
- T11-001 RESULT (jobs 94-96, 1.6 min): KILL. lane+pf chain +20..33%, AR x1.3-1.9, lane in-window x2.3-3 slower; net negative at every N/S. Write-up agents/leader-11/T11-writeup.md. Recommend close.
