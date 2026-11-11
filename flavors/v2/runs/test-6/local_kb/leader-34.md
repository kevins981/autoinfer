# LOCAL_KB leader-34 (T34, explore)

## Cards (2-3 lines)
Cards: "Four one-way lanes" + "Coordination" pointed at the AR transport itself; "Imbalance": in exposed mixed-step ARs the SMs are idle,
so SM-driven host reads (which contend at decode) are free there. Others (texture units, VMM, unified mem, Python) -> nothing actionable.

## Direction candidate (2026-11-10): prefill-size AR transport at its duplex floor
- Exposed mixed-step AR = 4.0 s / 42 s per 200 prompts (T32-5 table: 0.29/0.42/0.91/2.0/0.41 s by pt class).
- Champion big-step DMA AR at 1100 rows: 0.8 ms, of which ~0.4 ms fill/drain (D2H alone at start, H2D alone at end; leader-30.w1/attr RESULT §3); per-direction 21 GB/s.
- Duplex floor 11.2 MB / ~22 GB/s ~ 0.5 ms; compressed (0.78) ~0.4 ms.
- Hybrid CE-write + SM-read lost at DECODE (T16-001: ~20 us CE-branch latency in graph); never measured at prefill sizes.
- spar T34-1: "excess latency = necessary peer wait" is ASSUMED (no breakdown). SM reader rate with both ranks active: to test.
- Plan stage 0: microbench both ranks, R=128..1100, hybrid vs champion path, bitwise, graph, globaltimer boundaries producer-end -> consumer-ready.

## Stage 0 results (jobs s3/s4, leader-34.w1, bank/T34-001)
- Floors (rank 0, per AR, us): R=1100: CE D2H alone 441, SM host read alone 456, both (duplex) 633. R=512: 212/211/326. Duplex nearly free (~25 GB/s each).
- Hybrid with every CTA polling host flags: 1233 (16 CTAs) .. 4573 (64 CTAs) at 1100 -> worse than champion DMA ~800 (leader-30 trace, NOT same job).
  Grows with poller count: polling storm (worker hypothesis, untested) -> attempt failure, not limit.
- Forecast correction: duplex floor 633 us at 1100, not ~450-500 I assumed: raw-hybrid ceiling only ~-21% of big-step AR. Codec needed to approach -35-40%.
  Re-estimated payoff if works: ~+3-4.5% incl waves -> likely BELOW the 5% bar unless codec/fusion beats napkin. Be honest in proposal.
- Safety: crash 345 was in A_* arms (wait-value / freed pointer), H/F arms clean. Never run unbounded waits in benches.

## Stage 1 step 1 (job s7, T34-002): premise falsified
- Same-job exposed 16-AR chain, us/AR R=1100/768/512/256/128: H1 660/462/320/180/106; champion DMA 699/476/329/183/113; split 705/491/333/189; hostar(0.78) 620/446/303/161/89.
- Champion DMA within ~11% of raw duplex floor (631); in-situ ~800 was skew + lockstep chunks. H gain <=0.8% wall < gate. Did not integrate.
- spar T34-1 end: "scheduling ceiling ~2%" is ASSUMED (waves 131->116 achieved; 116->90 gap unexplained). Codec gap 620->543 recoverability assumed.
- Recommended closing at resync; ~68.6 GPU-min to return.
