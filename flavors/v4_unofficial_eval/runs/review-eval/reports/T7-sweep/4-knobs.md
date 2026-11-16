# T7 sweep, step 4: knob by knob (gains vs the champion's measured 895.4 tok/s; decode ~ 65-70% of time)

## Speculate (serial edge t -> t+1; Gate 1 holds because weights lack i; Gate 2: token id)
| Site | Napkin | Verdict |
|---|---|---|
| S1 chain length k=2 -> 3 | p3 ~ 0.567 x (0.567/0.782) ~ 0.41 -> tokens/step 2.34 -> ~2.75 (+17%). Step cost: +1 draft step (~1.0 ms), verify rows 48 -> 64 (GEMMs ~+0.4 ms, AR bytes +33% ~+0.5-1 ms, GDN 4 rows +0.4 ms, attn/sampling +0.3) ~ +2.6-3.1 ms on ~27-29 ms (+10%). Net decode +6%, overall ~+4%. Break-even p3 ~ 0.25. Fits the champion code: DeltaLog kMaxLogRows=3 (T=4 OK), HMAR <=96 rows (64 OK), GEMM configs at M=64. Flag only. | **#1, testing now** |
| S2 k=4 | T=5 exceeds kMaxLogRows -> old GDN kernel (loses DeltaLog's -1.8 ms); M=80; p4 ~0.3 | needs kernel change (kMaxLogRows=4); after S1 |
| S3 tree drafts (top-2 at depth 1) | rows 16 x 4-5; GDN tree verify needs branch states (in-place S: framework 02 §4.4 trap) | large build, out of time |
| S4 better drafter (trained) | allowed by ruling; needs training data/time | out of scope for 1 h |

## Keep / Recompute
| GDN state: DeltaLog already (Row B). Row C (keep raw k,v suffix) for the speculative rows only: already effectively what DL does. | done by champion |
| Materialize per mixed step (~1.5 ms): skip for requests not in the mixed part | small (~0.5%), code change | later |

## Rewrite
| GDN dual form for M=3 rows: (QK^T)V for the 3 verify rows instead of 3 sequential state updates: still must read S once -> bytes unchanged | ~0 | dropped (bytes-bound on S read) |

## Split / Place
| TP=2 -> PP=2: removes 128 ARs/step (~6 ms) but needs >=2 microbatches in flight; with c=16 and 2 stages each stage reads half the weights per microbatch of 8 -> step time ~ same weight bytes, AR replaced by 1 transfer; lm_head asymmetry (05 §4.1). Napkin up to +15-20% on decode, but a large engine change with bubbles at mixed steps | high gain, high risk; not buildable in 1 h |
| Asymmetric split for lm_head | only matters under PP | with PP |

## Window
| AR overlap with next GEMM weights (prefetch): champion does L2 prefetch already. Split r into 2 halves and overlap AR(R1) with GEMM(R2): framework 06 §3 says loses at small N (weights read twice) | loses at M=48 (2x weight reads 13 ms vs AR 6 ms) | dropped |
| Overlap draft step with next verify: serial dependency | invalid | dropped |

## Ranking (napkin): S1 (+4%, flag) > PP=2 (+15%?, weeks) > S2 (needs kernel) > materialize trim (+0.5%).
