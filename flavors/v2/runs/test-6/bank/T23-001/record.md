team:         T23
builds_on:    BASE (analysis of T14-004's in-situ nsys B1; no code change)
performance:  "n/a: measurement only (CPU)"
summary:
  hypothesis: the drafter (draft steps), small mixed-step glue and Marlin slab padding hold sizable removable per-step overhead on the champion.
  change:     none. CPU attribution of agents/leader-14.w1/insitu/B1/profile.sqlite by leader-23.w1; scripts + tables in agents/leader-23.w1/W1-1/.
  results:    Observed: decode step span 23.5 ms (nsys), real inter-step gap median 0.74 us (leader-19's "3.1 ms gap" = one 1.5 s outlier / 495 steps). After the last target layer 2.74 ms/step: target lm_head 1.01, draft lm_head 2x306 us (1.16x byte roof), MTP Marlin 0.41 (~roof), drafter host collectives 0.25 (all waiting on rank 1; rank 0 never waits; skew = GPU0 ~15% slower on ~35 tiny sampling kernels), fc 0.15. Small mixed steps (pt<256, n=27, mean M~132): 67 ms vs 23.5; excess Marlin +20.5, comm +16 (transport model), 587 vs 46 glue launches +2.4 ms, chunked GDN +2.3, in-span idle +3.1. Marlin per layer ~210 us per 64-row slab (staircase).
  conclusion: Interpretation: the drafter is at its floor (<=0.3% left); the small-step glue/idle (~5 ms/step on ~220 steps, ~1%) is the only sizable non-Marlin item; slab padding sized in T23-002.
validity:     no change.
