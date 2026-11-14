# T58 sweep (leader-58): summary

Framework: framework/ (01-07). Champion analysed: T52-005 (1212.4 tok/s official).

Files, in order:
1. [1-inputs.md](T58-sweep/1-inputs.md): model, hardware, workload, contract, and the champion's design read from its code, mapped to knobs.
2. [2-graph.md](T58-sweep/2-graph.md): data/compute nodes, indices, edges, and what lacks which index.
3. [3-calibration.md](T58-sweep/3-calibration.md): measured constants and napkin vs. measured.
   - **Sources:** jobs 485, 490/491 and 494, plus GPU-side reanalysis (no GPU).
   - **Decode step:** 24.6 ms measured vs. ~22-26 napkin.
   - **Mixed step:** 37 ms + ~0.15 ms per prefill row (GPU-side).
   - **Whole trial:** napkin within 5% of the 75.9 s measured.
4. [4-knobs.md](T58-sweep/4-knobs.md): the sweep, knob by knob, with napkin gains and pruning reasons. The addendum covers the mixed-step decomposition.
5. [5-history.md](T58-sweep/5-history.md): each candidate marked built/tested/untested, what the history knew that the sweep missed, and corrections to files 1-4.
6. [6-ranking.md](T58-sweep/6-ranking.md): the surviving candidates, ranked.

Headline:
- **Calibration:**
  - A decode step is 24.6 ms. GEMMs take 12.5 ms and run at the weight-byte floor, helped by the champion's prefetch during the all-reduce. The 130 host-staged all-reduces take 7.3 ms (30%). GDN takes 2.0 ms, the drafter 1.8, lm_head 1.0, and norms and attention 1.4.
  - Mixed steps take ~1/3 of GPU time. A small mixed step costs ~18 ms more than a decode step: its GEMMs leave the decode path (M > 96) and lose the all-reduce's prefetch.
  - The GPU is 98% busy.
- **My top two napkin candidates were already closed by the run:**
  - All-reduce latency (KB109/78/74/130).
  - f16-accumulate FP8 prefill MMAs (overflow; T38).
- **Ranked untested candidates, all ≤ ~1% each:**
  1. M1: prefetch during mid-size all-reduces. One env var; in test.
  2. M2: a mid-M FP8 GEMM.
  3. Lower the wave threshold by capturing the cores.
  4. Load-adaptive draft depth in the drain.
- **Framework gaps found:**
  - The link node has no latency or duplex model; "Link" here is SM or copy-engine access to host memory.
  - GEMMs below the ridge don't run at DRAM bandwidth. The ridge here is ~172 rows, not 333.
  - F depends on accumulator semantics, which the contract limits.
  - No resource for host time or request admission.
  - Capacity is coupled to speculation depth through GDN state pages.
  - Overlap is zero-sum on this stack, so the window rule max(F, BW, Link) is optimistic.
