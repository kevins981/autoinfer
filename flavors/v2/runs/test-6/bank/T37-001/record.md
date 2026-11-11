team:         T37
builds_on:    T31-003 (+ a python-only step-time logger, env T37_STEPLOG; no change to the computation)
performance:  "unofficial: no change intended (measurement record); 1099.7 tok/s on 300 prompts with the logger on (job 368)"
summary:
  hypothesis: none; a measurement campaign to find where the champion's time goes (seed grant).
  change:     gpu_worker.execute_model appends (perf_counter, prefill tokens, decode rows) per step when T37_STEPLOG is set.
  results:    Observed (jobs 366-371; scripts and outputs in agents/leader-37/{scratch,out}):
    - 300 prompts c=16 (368): 2120 steps, pure-decode step median 23.32 ms (p10-p90 23.0-23.6); extra time caused by
      prefill 15.6 s of 65.1 s (24%); fit over 185 prefill events: extra = 15 ms/event + 166 us/prompt token;
      prompts of 49-128 tokens cost a flat ~29 ms.
    - nsys node/graph traces (366/367): decode step GPU idle 0.1-0.2 ms; kernel time per decode step: e4m3 GEMM 5.5,
      Marlin 5.0, bs_gemm 3.9, host all-reduce 5.5, GDN replay 1.5, lm_head 1.0 ms. A step with a 60-token prefill:
      65 graph launches + ~1,130 eagerly launched kernels, GPU idle ~24 ms of 62 ms (graph-level trace).
    - draft depth (369, 150 prompts each): k=1 20.89 ms/step at 32 rows, acceptance length 1.784, 886.9 tok/s;
      k=3 25.07 ms/step at 56 rows, acceptance 0.774/0.552/0.400 per position (length 2.726), 1015.9 tok/s, but only
      14 sequences run concurrently (TTFT p50 490 ms).
    - max-num-seqs 32, c=32 (370): at most 19 sequences run concurrently (57 rows, 25.16 ms/step).
    - k=3 with --kv-cache-memory 5.6 GiB (371): CUDA OOM at startup (49 MiB free). k=2 reference in the same job,
      200 prompts: 1066.0 tok/s.
    - CPU: FP8 weight bytes have ~6.5 bits/byte entropy (layers 10-11, all matrices).
  conclusion: Interpretation: step time ~ 16.9 ms + 0.10 ms*rows + 0.83 ms*draft steps (fits k=1,2,3 within 0.15 ms);
    the fixed per-pass cost dominates. k=3's end-to-end loss is caused by the state-page pool seating 14 of 16
    sequences (pages per request grow with k), not by its step economics. Prefill has a fixed cost per event of ~15 ms.
    CORRECTION (after reading T32-003, pointed to by the manager): that cost is mostly GPU-side, not host-side. Kernels
    attributed by launch (job 366): a 104-row mixed step spends 19.3 ms in bs_gemm vs 12.5 ms GEMM in a 48-row decode
    step, 12 ms in all-reduces vs 5.9; removing the host launches alone was neutral in T32-003. Basis of proposal proposals/T37-1.md.
validity:     logger only; reads scheduler output, writes a file; off unless T37_STEPLOG is set. Not for official eval.
