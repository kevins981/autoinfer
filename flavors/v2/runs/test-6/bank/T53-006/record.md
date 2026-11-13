team:         T53
builds_on:    T50-002 kernels via T51's chain harness (agents/leader-51.w1/m1/bench_chain.py); no engine change
performance:  "measurement only: decode-megakernel ceilings at M=64 per GDN-proxy layer: small ops 19.5 us, GEMM
               boundaries ~0 (merging made it 20.8 us worse), GEMM vs DRAM floor ~33 us"
summary:
  hypothesis: Bound what a decode megakernel could win per layer: removing small ops, removing GEMM boundaries
              (merged weight streams), and GEMM inefficiency vs a pure DRAM stream of the same bytes.
  change:     agents/leader-53.w1/ceil/bench_ceil.py + job.sh. A copy of T51's harness logic:
              - 64 distinct layers (192 MB of weights each), 2 CUDA graphs of 32 layers, M=64.
              - VAR=a champion dispatch: qkvz/gdn_out/down t31b cfg 292, gate_up bs_gemm cfg 127.
              - Small ops and AR are spin kernels with the harness's durations: norm1 3, small 4, conv 4, quant 1,
                norm2 3.5, silu 2.5; AR 39/50 us.
              - GDN dummy = 12.5 MB copy + 37 us spin.
              - The harness has no separate split-K-reduce or elementwise ops, so (ii) removes the 6 small spins.
              - Standalone cold timings: read flush + cudaCtxResetPersistingL2Cache.
              - Merged GEMMs: qkvz+gate_up as one K=5120, N=25600 GEMM; gdn_out+down as one N=5120, K=11776 GEMM.
                Weights are fresh, same bytes. The config is picked by a cold standalone time among bs 127/27,
                t31 322 (A) and t31 292, bs 10/12 (B).
              - Floor: ms1ext stream_kernel (ld, U=8, 16 B step) over each of the 4 weights + GDN copy with no spin;
                128/256/512 blocks.
              Job 462, 0.1 GPU-min.
  results:    Observed (us/layer, medians of 5x(3x64 layers)):
              - (i) chain V1b (AR spins, cold): 387.4. (i0) V0b champion (AR with in-AR prefetch): 342.1.
              - (ii) small ops removed: 367.9. (iii) small ops + AR removed: 277.8.
                (iii') = (iii) with gate_up on t31b too: 365.9.
              - (iv) merged GEMM stream (A = bs 127, B = t31 292): 298.6.
              - (v) DRAM floor: 234.5 (256 blocks; 128: 234.7; 512: 235.1), i.e. ~0.93 TB/s over 217 MB.
              - Standalone cold, separate GEMMs: qkvz 51.2, gdn_out 24.4, gate_up 99.4 (bs 127; t31 322: 190.0),
                down 56.3. Sum 231.3.
              - Standalone cold, merged: A 173.5 (bs 127; bs 27 176.1; t31 322 288.8) vs 150.6 separate.
                B 74.0 (t31 292; bs 10 74.3) vs 80.7 separate.
              Differences:
              - Small-op ceiling i - ii = 19.5 us/layer (the spins sum to 18.0; ~1.5 of launch gaps).
              - AR spins (ii - iii) = 90.1.
              - Boundary ceiling iv - iii = +20.8 (merging is worse).
              - Floor gap iii - v = 43.3. That includes the 37 us GDN spin vs a ~27 us GDN copy in the floor, so the
                GEMM-only gap is ~33 us/layer (~240 vs ~207 us).
  conclusion: Interpretation: per 64-layer forward at M=64:
              - Removing small ops is worth at most ~1.25 ms (as modelled: spins without memory traffic).
              - GEMM boundaries already cost ~0 inside a CUDA graph: (iii) ≈ the sum of the standalone GEMMs +
                GDN + ~1 us per launch. Merging weight streams with the existing kernel families loses: the merged
                gate_up-like GEMM runs at 0.76 TB/s vs gate_up's 0.90, and only 3 configs were tried.
              - The GEMM inefficiency vs a pure stream is ~33 us/layer (~2.1 ms/forward, ~16% over the stream).
                Most of it is gate_up (99.4 vs ~92 at the stream rate) and qkvz/down (~51/56 vs ~45/48).
              - The big items remain the AR (modelled at 89 us/layer as spins) and the GDN compute.
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
