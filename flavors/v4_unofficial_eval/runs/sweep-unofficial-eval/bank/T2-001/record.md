team:         T2
builds_on:    BASE
performance:  never evaluated (superseded by T2-002 after HEAD moved to T1-001; job 7 cancelled)
summary:
  hypothesis: mixed (prefill-containing) steps above 96 tokens run fully eagerly (server log: max_prefill_cudagraph_capture_tokens 96, compilation mode NONE) and are CPU-dispatch-bound: graph-level trace of BASE (agents/leader-2/prof_graph, GPU job 2) shows mixed steps = 38% of time, small ones 133 ms with ~53 ms GPU idle + rank-skew waits vs ~55-65 ms of GPU work. Capturing prefill CUDA graphs up to max_num_batched_tokens (2048) should cut those steps to ~GPU time: napkin -8..10% total time (reports/T2-sweep/3-calibration.md, 4-knobs.md W1).
  change:     serve args only: --compilation-config '{"max_prefill_cudagraph_capture_tokens":2048}' (Model Runner V2 option; default on SM100 is 8192, elsewhere the shared 96).
  results:    pending
  conclusion: pending
validity:     Execution mode only (CUDA graph replay of the same kernels, breakable graphs as in BASE's decode path); math unchanged. Memory: more/wider graphs -> run with --long. No benchmark-specific behavior (sizes are a generic token grid up to max_num_batched_tokens).

unofficial #1 (2026-11-15T05:19): FAILED at exit 130 (unofficial/1/)
