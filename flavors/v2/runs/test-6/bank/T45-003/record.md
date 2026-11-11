team:         T45
builds_on:    T37-002 (kernels as in T31-003/T37-002; harness from T35-003)
performance:  "microbench only (stage 1 of proposal T45-1): kill rule hit, 86 us/row/pass marginal GEMM cost from M=64 to 96 (rule: <=40)"
summary:
  hypothesis: Below the tensor ridge, extra rows in a decode pass are nearly free in the GEMMs (<20 us per row per pass up
              to ~100 rows by the floor argument), so prompt rows riding in decode passes would cost only their AR
              (~0.07 ms) instead of 0.166 ms in prefill events. That would make KB64/KB108's negatives artifacts.
  change:     None to the server. T35's standalone 64-layer GDN-layer chain (agents/leader-35.w1/ms1, real t31 e4m3
              GEMMs for qkvz/gdn_out/down and stock Marlin fp8 gate_up, 64 distinct random-weight layers = 12 GB, CUDA
              graphs, AR spin 39/50 us with the champion's in-AR prefetch), run at M = 48/64/80/96 (t31 has no M>96
              path: M=112/128 raised "M out of range (1..96)"). Copy in bank/T45-003/bench_m.py, job.sh, res_M*.json.
              GPU job 408, GPU0, 0.8 GPU-min.
  results:    Observed (champion-like variant V0b, us per layer; per-GEMM means from the profiler):
              M   layer   qkvz  gdn_out gate_up  down   sum GEMM
              48  356.8   40.6  21.6    94.9     52.1   209.2
              64  368.6   41.5  21.5   104.1     53.8   220.9
              80  404.9   42.3  24.5   131.0     59.7   257.5
              96  411.6   42.6  24.6   131.5     59.9   258.6
              Marginal per row per pass (x64 layers): 48->64 47 us; 64->80 145 us; 80->96 27 us; 64->96 86 us.
              Of the 64->96 step, gate_up is 27.4 of 43.0 us/layer (55 us/row/pass); the three t31 GEMMs 21 us/row/pass.
              Estimate written beforehand: <20 us/row/pass. Serial no-prefetch variant (V1b) shows the same step.
  conclusion: Interpretation:
    - The band cost is a step, not a slope: crossing 64 rows costs ~2.3 ms per pass at once (one more m-block), and rows
      80->96 are then cheap. Adding 32 prompt rows to a 64-row pass costs ~86 us/row of GEMM + ~70 us of AR = ~0.156
      ms/row, about the 0.166 ms of a prefill event. Piggyback saves ~nothing with these kernels: KB64/KB108 are
      consistent with this.
    - The step is mostly physics for a bf16-MMA gate_up, not an implementation artifact: gate_up at M=96 is 17.1
      GFLOP/rank, 117 us at Marlin's bf16 MMA peak (~146 TF) vs 131.5 measured (89%); its weight stream floor is ~91
      us. So gate_up's ridge with bf16 MMA is ~M=75: the "sub-ridge slack" I assumed (computed at the FP8 MMA rate)
      does not exist for the largest GEMM, 40% of the bytes.
    - The slack exists only if gate_up runs on e4m3 MMA (2x rate, ridge ~150 rows): then ~60 us of compute at M=96
      sits under the ~91 us stream. The champion keeps Marlin for gate_up because e4m3 gate_up did not win at M=48
      (KB98/KB101). Whether an e4m3 gate_up holds ~95 us at M=96-128 is untested, and is what any stage 2 must
      measure first. The t31 GEMMs also need an M>96 path.
    - Caveats: synthetic weights; GDN-layer proxy for all 64 layers (no attention layers); stock Marlin (not T6's slab
      policy) for gate_up; AR modelled as fixed spins (its per-row cost is taken from KB104/KB109, not measured).
validity:     Microbenchmark only; nothing to review for serving.
