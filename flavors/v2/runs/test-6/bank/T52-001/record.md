team:         T52
builds_on:    T50-002 (microbench only; no engine change)
performance:  "microbench negative: fused AR2+norm+quant+qkvz kernel 103.5 us vs champion slots ~88.5-90.4 us (NL=4 chain, rank 0; jobs 440/443/444, 0.6 GPU-min)"
summary:
  hypothesis: one persistent kernel doing [host-staged AR -> residual add + RMSNorm -> quant_perm -> qkvz t31b GEMM], with GEMM CTAs
              streaming their own weight slab into L2 (evict_last) during the AR, costs ~max(AR, stream) + warm tail (~76 us est.) vs ~100.
  change:     agents/leader-52.w1/fused.cu (128-CTA co-resident grid = t31b qkvz cfg; first 2G CTAs run the champion hostar AR code
              mode 20 + per-chunk add/norm/quant with per-z ready counters; others stream slab then wait; copied t31b main loop),
              chain harness chain52.py/.sh (T31's harness at M=64 with champion kernels). Notes: agents/leader-52.w1/NOTES.md.
  results:    Observed: correctness 100% bitwise vs the champion chain (AR sum, residual, Aq/As, GEMM output).
              (b) qkvz t31b M=64 warm 32.7 us, cold 45.5; in the champion chain 34.1-34.5 (hint prefetch already ~warm).
              (a) AR inside fused kernel (last peer flag / sum / norm+quant ready, us): AR+norm only 44-48/52-53/56; +full 42 MB stream
              +10 us (16:2), +21 (8:4), +9.5 (32:1); stream capped at ~11-19 MB free (<=1 us). Champion hostar kernel alone 50.4.
              (c) chain slot: champion 50.0-51.3 + 2.7 + 1.7 + 34.1-34.5 = ~88.5-90.4 us; fused best 103.5 (16:2, no stream),
              capped stream 104.6, stream-until-ready 109, 32:1 107, 8:4 114-124. Layer 325-344 vs 306-308 us.
              In-chain trace (16:2): ready at 64.6 us, GEMM go 67.5, end 102.
  conclusion: Interpretation: whole-row consumer fusion is bounded by AR (~50) + the t31b warm floor (~30); the champion already gets
              AR + 34.5 + 4.4 with free hint prefetch, so <=5-9 us is available and HBM streaming during the AR stretches it by more.
              My premise (qkvz 49 us in situ from a node-traced nsys) did not hold in the chain (34 us). This attempt failed; the
              cause is that the GEMM waits for the whole row and is ~30 us post-flag. Next variant: K-chunk cut-through consumer
              (GEMM consumes K-blocks as AR chunks land; 1/rms deferred per the 2026-11-11 ruling) so the AR read half overlaps the GEMM.
validity:     microbench only; exact (bitwise). Not for official eval.
