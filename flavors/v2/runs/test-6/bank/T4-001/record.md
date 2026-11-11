team:         T4
builds_on:    T1-042
performance:  "microbenchmark only: premise FAILED. Hiding a decode AR under its consumer GEMM saves at most ~5-8 us/AR (bar was 15)"
summary:
  hypothesis: Each TP all-reduce at decode (M=48, ~46 us) can run beside its consumer GEMM (made possible by deferred RMSNorm scaling, manager ruling 2026-11-09), saving ~30-40 us/AR (~3.5 ms of a 23.8 ms step).
  change:     T1-042 + MARLIN_SMS grid override (from T2-001) in marlin.cu. 2-process microbench by leader-4.w1 (jobs 24-27, ~1.5 GPU-min): agents/leader-4.w1/{ovl.py, ovl2.py+stamp.cu, ovl3.py, job.sh}, logs out_*_r{0,1}.log. 16 layers = 32 ARs, real FP8 Marlin (random weights), per-rank shapes, CUDA graphs, 6 interleaved reps (agree <0.3%).
  results:    Observed (ms per 16-layer chain, r0/r1):
    - A_pf champion path (host AR + compression + L2 prefetch) 5.019/5.018; A_nopf 5.594/5.600; N_full (no AR at all) 4.240/4.107.
    - B (AR on a side stream beside a consumer with grid 128-f; consumer doesn't wait, so the math is wrong; this is the ceiling): f=8 5.105, f=16 4.944, f=24 4.922, f=32 4.850 (r0). B-DMA (copy-engine AR) grid 120: 4.818.
    - Per-AR saving vs A_pf: -1 (f=8) .. +5-8 us (f=32 / DMA). Estimate was 30-40 us.
    - C (grid 128-f, no AR): only +1-5 us/AR over N_full, so the smaller grid is cheap.
    - Timeline (job 25): the AR starts +0.8 us after the producer, but it runs 51 us alone vs ~100 us beside gate_up. GEMM gate_up 115 -> 139 us at f=8, 117 at f=16. qkvz/down (~50 us) are too short to cover the slowed AR.
    - Ceiling of removing the AR entirely: A_pf - N_full = 24-28 us/AR (~3.2 ms/step). Prefetch already recovers ~18 us/AR (A_nopf - A_pf).
    - First-chunk latency (host AR, 48 x cols): 1/8 width 12.1 us, 1/4 16.2 us, full 39.4-40.1 us compressed / 46-48 uncompressed. Estimate 10-15 us: matches.
  conclusion: Interpretation: on this machine an AR doesn't overlap a memory-bound GEMM. Its PCIe round trips slow ~2x under Marlin's HBM traffic, and the GEMM slows too (the same zero-sum picture as KB59, now for comm vs GEMM). The direction "hide the AR under its consumer" is dead as formulated. The best case, hiding only the gate_up AR, is <= ~1.5% of a decode step. Removing ARs (not hiding them) is worth ~25 us/AR, but every route to fewer ARs found so far costs memory we don't have.
validity:     measurement only; no served change.
