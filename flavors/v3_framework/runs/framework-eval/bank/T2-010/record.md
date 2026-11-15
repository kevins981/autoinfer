team:         T2
builds_on:    T2-009
performance:  "smoke inconclusive (1 pair, bench 100, -0.8%, within noise); no ABBA run"
summary:
  hypothesis: Decode W8A8 GEMMs at M=48 looked to be at 67-78% of a 1008 GB/s byte floor (nsys). The configs might be
              suboptimal, especially given T3-003's prefetch.
  change:     (worker leader-2.w1, ws t2-mm) Deterministic two-pass split-K in the Triton W8A8 block GEMM. Prefetch-
              optimal configs for gate_up/in_proj/qkv at M buckets 16/32/48. Gate VLLM_T2_GEMM_PF_CFG.
  results:    Observed:
              - Microbench, cold, weights rotated so L2 cannot hold them (jobs 103/105/107): current configs already
                optimal (within 0-1.3%). The achievable streaming read is 850-890 GB/s, not 1008, and the GEMMs match
                a plain read kernel. In situ, GEMMs are slower than this micro (gate_up 131 vs 100 us), so the gap is
                context, not tiles.
              - With 40 MB prefetched: gate_up -9%, in_proj -24% (split-K S=3), qkv -13%.
              - Split-K correctness (job 111): rel err <= 3e-3, deterministic.
              - Server smoke (bench 100, 1 run/arm): gate 1 842.9 vs gate 0 849.4 tok/s; acceptance equal.
  conclusion: Interpretation: no end-to-end signal. The in-situ prefetch doesn't leave the whole weight resident the
              way the micro assumes. Not worth a 10+ min ABBA to resolve ±1%. Revisit only if the prefetch is
              extended or retimed.
validity:     fp reordering only (split-K fp32 partials, fixed-order reduce). Not proposed for official.
