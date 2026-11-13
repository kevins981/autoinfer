team:         T52
builds_on:    T52-005 (chain microbench; dma_staged_ar.cu variant in workspaces/t52-w2 only)
performance:  "chain microbench negative: copy-only ordering (WD) 1001/1705 vs WP 973/1670 us/layer at M=384/736 (job 479, 0.2 GPU-min)"
summary:
  hypothesis: the ~100-300 us per-AR peer lag in WP is coupling created by link priority (each AR waits for the previous AR's completion, which
              includes the peer's D2H); ordering only the D2H copies (FIFO, no completion waits) removes it: est. M=736 1664 -> 1550-1600.
  change:     WD arm in agents/leader-52.w1/m2 (m2w): B's ARk(L) D2H waits on A's ARk(L) D2H-done, A's AR1(L) on B's AR2(L-1); per-lane H2D kept
              (one shared H2D stream would queue B1 behind A's MLP). Deadlock reasoning in NOTES.md; all waits bounded.
  results:    Observed (NL=8, median of 7, us/layer; M=384 / M=736): J 1156/1904, WP 973/1670, WD 1001/1705, floors Jn 851/1390, Wn 787/1317.
              WD bitwise = WP. Lag p90 halves (rank0 300 -> 137, rank1 214 -> 138 us) but copy/kernel overlap on rank 0 drops 6.5 -> 4.8 ms and
              copy-only time rises 1.24 -> 2.26 ms over 8 layers. Eager CPU issue in chain: WP 325/421 us per wave-layer, J 624/790 per layer
              (server eager wave steps ~0.9-1.1 ms per wave-layer -> ~0.5-0.7 ms is vLLM-side overhead).
  conclusion: Interpretation: the lag is a symptom; the completion waits were spacing copies under compute. The limit is cover per AR (T30's
              finding), not a separate coupling lever. Closed; WP kept.
validity:     microbench only.
