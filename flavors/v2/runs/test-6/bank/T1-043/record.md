team:         T1
builds_on:    T1-042
performance:  "unofficial: KILLED: lean ABBA (job 181, 150 prompts) B mean -0.6% vs the single A arm (864.0 vs 856.5/860.7); 4A cut by the timeout"
summary:
  hypothesis: Folding residual add + gemma RMSNorm (129 per decode step, 3.6 us each in-server) into the host-AR reader epilogue saves ~2-2.8 us per AR, ~+0.8-1.0%.
  change:     ws leader-2.arnorm (w23), off with VLLM_HOSTAR_NORM=0.
              - hostar kernel gets M/2 extra norm blocks (2 rows each) that wait on a reader-done counter, then compute h = f32(s) + f32(res), a fixed-order sum of squares, rsqrt.approx, y = bf16((h*rstd)*(w+1)), and write res' and y.
              - used for hostar ARs <= 1 MB with gemma-form norms (128 layer norms, the final norm, MTP); the first layer stays on the old path.
              - Includes a fix for a prefetch-stride bug found during this work: prefetch blocks counted the norm blocks in their stride, under-covering the prefetch range. Outputs were unaffected (hints only).
  results:    Observed:
              - Unit test (job 179/180) vs hostar + flashinfer: res' bitwise; y <= 1 ulp (13 of 1.72M elements); graph == eager; negative control (w instead of w+1) 245,630 off.
              - Microbench after the fix (job 180), us/AR AR+norm -> fused: M=16 36.3 -> ~33 (-3.8); M=48 45.2 -> 44.5 (-0.7); M=96 -0.4. Estimate was 2-2.8 us at M=48.
              - ABBA (job 181): tok/s 1A 864.0 | 2B 856.5 | 3B 860.7; 4A cut by the timeout.
              - gen_check 0/10 identical (fp reorder; divergence at tokens 42-272); gsm8k50 88%.
  conclusion: Interpretation: the norm is small next to the AR tail it would hide in; fusing adds ~1.5 us of epilogue that eats most of the saving. No in-server gain. Killed.
validity:     correct within fp reordering (res' bitwise, y 1 ulp); not proposed for official eval.
