team:         T1
performance:  "unofficial: +2.2% (645.2 vs BASE official 631.4 tok/s, 400 prompts c=16), within noise"
summary:
  hypothesis: steps containing a prefill (~35% of wall time) run eager because the prefill CUDA-graph
              grid stops at 96 tokens on non-SM100; raising it to 2048 should remove CPU launch overhead.
  change:     no code; serve flag --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'
              (serve_args.txt). From BASE.
  results:    Observed (worker leader-1.w1, 1 GPU job, lib.sh bench 400 16; logs in
              workspaces/leader-1.prefillgraph/out/run1/):
              - server log: prefill graphs PIECEWISE 31 sizes (1..2048) + FULL 9; decode FULL 11;
                breakable CUDA graph auto-enabled. Capture 10 s, 2.07 GiB per rank (BASE value not measured).
                KV 267,721 tokens (fixed by --kv-cache-memory). Ready after 60 s.
              - 645.2 tok/s; TTFT p50 223 / p99 1420 ms; TPOT p50 22.7 / p99 46.95 ms; ITL p99 291.8 ms;
                MTP acceptance 2.343. GSM8K n=50 88.0% (+-4.6; not comparable to n=250).
              - 128K test: 128000-token prompt, /v1/completions max_tokens 64 -> HTTP 200, finish "stop",
                sensible text; server healthy after. No errors.
              - worker's pre-run estimate: ~655 tok/s (+4%).
  conclusion: Interpretation: gain is within the ~3% noise (single run, no same-job BASE). The prefill
              steps are therefore NOT dominated by CPU launch overhead in the real (non-nsys) run; they
              look GPU-bound (Marlin + all-reduce + rank skew), consistent with adjacent_field's M1.
              Graphs are PIECEWISE (attention/GDN pieces still eager), so a FULL mixed graph was not tested.
              ITL p99 ~292 ms stays: prefill steps still stall decodes.
validity:     leader-1 review: empty diff; flag only changes CUDA graph capture sizes (same kernels/math).
              Memory change -> 128K test done and passed. OK by task_definition. No official eval requested
              (below noise); candidate to stack on later changes if it costs nothing.
