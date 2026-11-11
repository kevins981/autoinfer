team:         T2
builds_on:    T1-042 (workspace t2-m1 BUILT, every T2 env unset = champion)
performance:  "measurement only: critical-path anatomy of the champion's decode and mixed steps (supersedes T2-005)"
summary:
  hypothesis: Mixed steps (~39% of wall) hold a non-GEMM lever worth a direction (>= 5%).
  change:     none. nsys TRACE=node, 64 prompts c=16, champion serve_args (job 22, by leader-2.w1); replay/flush kernels and the Marlin tile confirmed in kern_sum. Exclusive critical-path attribution on rank 0 (priority marlin > hostar > GDN > attention > norm/act > lm_head > memcpy). Scripts and data: agents/leader-2.w1/anat/{prof2, cp_2.py, q*_2.py, q4.py}; tables anatomy_tables.txt.
  results:    Observed (median ms per step, rank 0):
    | family | decode | mixed pt<150 | pt 150-399 | pt>=400 |
    | period | 23.8 | 76.0 | 91.1 | 144.2 |
    | Marlin | 13.6 | 33.0 | 84.3 | 157.7 (2 streams) |
    | host-AR wait | 5.9 | 10.2 | 11.1 | 29.9 |
    | AR memcpy exposed | 0 | 0 | 7.5 | 12.8 |
    | GDN | 2.0 | 4.4 | 5.9 | 8.5 |
    | norm/act/elementwise | 0.4 | 3.0 | 3.6 | 5.5 |
    | attention | 0.8 | 1.6 | 2.2 | 3.4 |
    | lm_head/other | 1.0 | 1.2 | 1.3 | 1.5 |
    | idle in span | 0.1 | 1.7 | 4.9 | 4.7 |
    Steps per 400-prompt trial (x6.25): decode ~3150, mixed ~150 / ~50 / ~119.
    - GPU idle between steps = 0 for every step type; the ~18 ms "outside span" of small mixed steps is host run-ahead (the GPU runs ~25 ms behind the host; ~1790 launches per mixed step under nsys), not lost GPU time.
    - Host-AR per launch, rank 0: 41 / 84 / 144 / 285 us at M = 42 / 112 / ~180 / ~388 tokens (0.61 us/token + ~15.5 us): the transport floor at every size. Rank 1 is 20-40% longer (waiting for the slower-clocked GPU 0).
    - Mixed-step GDN decode rows use replay: 1.9 ms/step vs 1.3 ms in decode graphs.
    - Elementwise: act_and_mul ~8x above its byte floor but mostly overlapped with the other token-split half's Marlin; add_kernel = the DMA-AR combine (dma_staged_ar.cu:61), fusable into the following FusedAddRMSNorm; ~1100 small aten glue launches per mixed step (index_select in qwen_gdn_linear_attn.py:1878-1881, 1954-1955).
  conclusion: Interpretation: every mixed-step lever is small. Credible exact items sum to ~1.0-2.2 s of a ~97 s trial (+1-2%), each below noise alone: the DMA-add fusion 0.3-0.5 s, silu-mul into the Marlin epilogue 0.3-0.5, index glue 0.15-0.3, DMA tail 0.3-0.7, in-span idle <= 1.3. AR is at its transport floor; prefill Marlin at ~85% of its microbench peak. Decode steps (~75 s) dominate, and their parts are at measured floors (KB45, KB56). Decode floor estimate: ~12 ms weights + 3.8 ms AR bytes + 2.1 ms AR latency + 2.0 GDN + ~2.3 other ≈ 22 ms vs 23.8 measured.
validity:     measurement only.
