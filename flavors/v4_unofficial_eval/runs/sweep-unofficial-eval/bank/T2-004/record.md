team:         T2
builds_on:    T1-004
performance:  unofficial: PASS +4.49% vs HEAD T1-004 (695.5 / 691.3 vs 663.6; 200 prompts); 128K pass
summary:
  hypothesis: mixed (prefill-containing) steps above 96 tokens run fully eagerly (server log: max_prefill_cudagraph_capture_tokens 96, compilation mode NONE) and are CPU-dispatch-bound: graph-level trace of BASE (agents/leader-2/prof_graph, GPU job 2) shows mixed steps = 38% of time, small ones 133 ms with ~53 ms GPU idle + rank-skew waits vs ~55-65 ms of GPU work. Capturing prefill CUDA graphs up to max_num_batched_tokens (512) should cut those steps to ~GPU time: napkin -8..10% total time (reports/T2-sweep/3-calibration.md, 4-knobs.md W1).
  change:     T1-004 (HEAD) serve args + --compilation-config {"max_prefill_cudagraph_capture_tokens":512}. Rebase of T2-003 (its eval vs T1-001 cancelled, job 14, when HEAD moved to T1-004). T2-002 (cap 2048) OOMed at graph capture; cap 512 covers mixed steps up to 512 tokens (~75% of them in GPU job 2; larger steps are GPU-bound).
  results:    Observed: unofficial #1 PASS +4.49% (B 695.5/691.3 vs ref 656.3/671.0), MTP acceptance 2.353 (unchanged), p99 TTFT 1161 / TPOT 41.7 ms (both better than BASE's 1341/46.1), 128K request pass. Server log: 23 prefill piecewise graphs captured up to 512 tokens, graph memory 1.49 GiB (BASE 1.27).
  conclusion: Interpretation: graph-replaying mixed steps up to 512 tokens removes most of their CPU dispatch time; gain (+4.5%) is about half the napkin 8-10% (steps >512 tokens stay eager; part of the measured idle may have been nsys overhead). A cap above 512 OOMs at capture next to the FP8 draft head (T2-002); 1024 untested.
validity:     Execution mode only (CUDA graph replay of the same kernels, breakable graphs as in BASE's decode path); math unchanged. Memory: more/wider graphs -> run with --long. No benchmark-specific behavior (sizes are a generic token grid up to max_num_batched_tokens).



unofficial #1 (2026-11-15T05:44): PASS: +4.49% vs HEAD T1-004 (B 695.5 / 691.3 vs reference mean 663.6 over 2 starts, 0 run in this job; 200 prompts); 128K pass, p99 TTFT 1161 ms / TPOT 41.7 ms, MTP acceptance 2.353 (unofficial/1/unofficial.json)

official #1 (2026-11-15T06:09): 728.8 tok/s at c=16 (mean of 2 trials: 729.5 / 728.0, CV 0.1%) (+14.6% vs BASE), GSM8K 93.2%, MTP acceptance 2.344, TTFT p50 198 ms, TPOT p50 20.2 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
