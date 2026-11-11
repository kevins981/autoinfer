team:         T1
builds_on:    T1-040
performance:  "OFFICIAL #1 934.1 tok/s (+0.5% vs T1-040 929.1; both trials above both of T1-040); unofficial: microbench -1.65 us/layer at N=16 with GEMM (145.7 vs 147.4; repeats agree to 0.1 us); bitwise == T1-040; no ABBA (below e2e noise)"
summary:
  hypothesis: After RED8 the GDN replay kernel is latency-bound (KB51). Hiding prologue and chunk-load latency would cut 0.1-0.5 ms/step.
  change:     ws leader-2.gdnlat (w23). GDN replay kernel mode 5, now the default (VLLM_GDN_RED8=5; 1 = T1-040, 0 = 32-lane):
              - batched prologue: all log/qkv/gate loads issued up front into registers;
              - warp-private cp.async staging: each warp copies its own 4 RED8 rows (the rows it reads), __syncwarp instead of a per-chunk __syncthreads.
              Same fp32 ops in the same order.
  results:    Observed:
              - Premise (job 167, clock64 stamps, N=16 no GEMM; us per CTA): prologue 8.64, dependent-hop chain 0.75, chunk waits 1.60 (5.74 with GEMM), compute 7.93, epilogue 0.83.
              - Bitwise (job 169): 0 differing hashes vs T1-040 at cap 8/0/6 (27,075 each). Negative control 14,891 differing.
              - Microbench (us/layer alone / with GEMM):
                | config | T1-040 | mode 4 | mode 5 |
                |---|---|---|---|
                | N=16 cap 8 | 46.1/147.4 | 45.0/146.3 | 44.9/145.7 |
                | N=16 cap 0 | 62.8/161.8 | - | 62.6/160.9 |
                | N=8 cap 8 | 19.7/124.3 | - | 18.6/123.3 |
              - Premise for P1a/P2 (hop chain < 1 us) failed: not built. q/k norm dedupe est < 0.3 us: not built.
  conclusion: Interpretation: the prologue is mostly the first-load latency under DRAM contention (all CTAs start together). Only ~1.6 us/layer is hideable by scheduling: ~0.08-0.16 ms/step. Banked because it is bitwise and free.
validity:     leader-2 reviewed: scheduling-only changes to the replay kernel. Warp-private staging stages exactly the rows the warp reads (row = warp + g*kWarps, g = lane/8 in 0..3), so there is no cross-warp race without the CTA barrier; double-buffer reuse is guarded by wait_all + __syncwarp. Same fp32 ops and order (bitwise-tested). Default-only Python switch. OK.

official #1 (2026-11-08T14:21): 934.1 tok/s at c=16 (mean of 2 trials: 934.0 / 934.3, CV 0.0%) (+46.9% vs BASE), GSM8K 93.2%, MTP acceptance 2.327, TTFT p50 172 ms, TPOT p50 15.7 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
