# T9 sweep, 4: knobs (compressed; gains vs the 29.0 ms decode step, decode ~65% of wall)
| # | knob / site | napkin | verdict |
|---|---|---|---|
| A | Speculate x[t+1]: chain depth k=2 -> 3 (flag; champion code supports 4 rows: DeltaLog kMaxLogRows=3, HMAR 64<=96 rows, cudagraph size 64 captured) | p3 ~ 0.567 x (0.567/0.782) ~ 0.41 -> tokens/step 2.343 -> ~2.75 (+17%); step +2.2-2.7 ms (+8-9%) -> decode +7-8% -> ~+4-5% overall. Risk: MTP layer reused for draft 3, acceptance may fall faster (vLLM warns) | TOP: flag only, test now |
| B | Speculate tree (top-2 at depth 1) | M grows to ~64-80, needs tree attention + GDN branch states (S lacks branch index -> per-branch state copies): large build | too big for the time left |
| C | Split/Place: AR 5.96 ms latency-bound: merge ARs (fuse AR+norm, or 1 AR per layer via sequence-parallel restructure) | -2..-3 ms/step -> +5-7% decode | multi-hour kernel work: not in 60 min |
| D | Window: overlap AR with next GEMM (weights don't depend on AR) | champion already prefetches weights into L2 during AR (T3-001) | mostly taken |
| E | Keep/Recompute: target lm_head 1.27 GB bf16 per step | fixed by contract (target lm_head stays baseline) | drop |
| F | Rewrite: drafter fp8 lm_head already; reduced-vocab drafter (ruling 2) | drafter lm_head ~0.8 ms x2 -> ~0.3 ms with 32K vocab: ~-1 ms/step (+2-3%), needs non-benchmark token-frequency data | candidate, more work |
| G | k=3 + reduced vocab drafter combined | A+F stack | later |
