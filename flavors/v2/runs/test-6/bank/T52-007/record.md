team:         T52
builds_on:    T52-005 (measurement only; test-only timers/cProfile in workspaces/t52-w2)
performance:  "measurement: eager wave-step host loop 115-119 ms/step (k=2, n=800-1281); GPU-bound at n>=~800 (job 480, 2.0 GPU-min)"
summary:
  hypothesis: the host cost of eager wave steps (~0.9-1.1 ms per wave-layer vs ~0.33-0.42 raw issue in the chain) has a few fixable culprits;
              removing them lets T30_WAVE_MIN drop from 768 toward 384.
  change:     T52_WTIME per-segment perf_counter in _t30_wave_layers, T52_CPROF cProfile of one wave step (rank 0); 40 prompts c=16.
              Files agents/leader-52.w1/srv/s4/, NOTES.md "(b)(ii)".
  results:    Observed: steady loop host 115-119 ms per wave step (first two 168/205); hostprobe host-to-fwd-end 146 vs fwd GPU 160 ms.
              GDN layer per wave-pass ~905 us: core 385, mlp 104, ar1 96, out_proj 87, in_proj 77, ar2 75, post_norm 41, in_norm 39; attention core 480.
              cProfile (1.45x inflation): DMA AR C++ issue 62 us/call (~26 driver calls); torch.cuda.current_stream 768 calls 16.1 ms; Python triton
              launches 528 calls 26.8 ms; linear wrapper stacks 512 calls 26 ms (1809 torch.empty); GDN core glue 96 calls 40.7 ms (T32 replay 73 us,
              gated norm 13 ms, conv update 5.2 ms); attention eager via breakable wrapper 49 ms.
  conclusion: Interpretation: no single culprit: ~1/3 GDN core glue, ~1/3 linear wrappers, ~1/5 AR issue. Fix routes (worker estimates):
              lean loop (stream handles, C++ AR issue, direct linears, core glue into the captured graph) 117 -> ~55-60 ms, 3-4 days; dense-segment graphs
              for wave rows 192-448 ~165 MiB + core work -> ~40 ms. Waves at MIN=384 need loop <=~45 ms. Expected value ~+1% e2e (unmeasured).
validity:     measurement only.
