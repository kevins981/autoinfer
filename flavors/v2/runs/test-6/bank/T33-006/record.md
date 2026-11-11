team:         T33
builds_on:    T33-005
performance:  "microbench: escape-free main decode cheap (L2-hot floor 64.7 vs 57.3 us decode-off) but exact correction of 2.67% escapes has a lower-bound floor ~118-121 us; killed (gate: <= 70 us)"
summary:
  hypothesis: removing escape handling from the main decode (7-exponent window, code 7 = zero in main) and adding escaped products back exactly (each product computed once in fp32, grouped within the same 128-K block) makes the 0.8925 code pay in the prefetched regime.
  change:     agents/leader-33.w1/enc4.py, ov7.cu (k7): main rebuild ~116 instr/lane/stage (vs ~330 in T33-005); correction measured as a lower-bound emulation (27 escapes/tile, token-parallel FFMA, A from smem with real swizzle; no list load/merge), not value-correct.
  results:    Observed (GPU0, jobs 341/343): escapes 2.67%; main stream 0.8925 (implicit variant), explicit list variant ~0.939 (computed). L2-hot floor: decode-only 64.7 (S4) vs decode-off 57.3; with correction emulation 120.9 (S4) / 118.3 (S3), pf48 118.0. Decode-only standalone (not exact): cold 92.6 / pf24 81.9 / pf48 72.3 vs raw 101.9 / 85.8 / 71.6.
  conclusion: Interpretation: the main rebuild is cheap; exact handling of scattered escapes (48-token FMA per escape, ~14 warp-instr each) costs far more than the bytes save. Escape-free windows need >= 9 exponents (4-bit codes: no saving). Tested exact codecs lose in the L2-prefetched regime; denser (entropy) codes would decode more expensively, untested. Not a proven hardware limit.
validity:     microbench only (validity of the correction grouping would have needed a manager ruling; moot).
