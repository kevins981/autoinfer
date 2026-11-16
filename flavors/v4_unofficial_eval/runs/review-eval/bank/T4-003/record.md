team:         T4
builds_on:    T3-007
performance:  "stopped after 1 start, -0.26%: 851.4 tok/s vs the champion's 853.6 (eval/1, GPU job 88, STOP_BELOW=853.6); TTFT p99 976 ms, TPOT p99 33.6 ms, acceptance 2.329, gate_latency pass. GSM8K / 128K not run (stopped)."
summary:
  hypothesis: T4-001's one-kernel recurrent GDN prefill improves small mixed steps (unprofiled, matched shapes: 54.2 vs 65.3 ms for one
              58-64-row prompt + 15 decoders) and loses above ~256 rows; restricted to prompts of at most one FLA chunk (64 rows, a
              structural boundary: no chunk parallelism below it, equal GPU time at T=64) it should keep only the gain: napkin from the
              T4-001 iteration logs ~30 such steps per 200 prompts x ~11 ms -> ~+0.6%.
  change:     by worker leader-4.w1 (workspace t4-rp2 = T3-007 + T4-001's kernel and hooks), with review T4-2's bugs fixed: strided
              state indices (stride passed to the kernel), initial-state load and final store only for idx > 0 (idx <= 0 = fresh, no
              store; NOT equivalent to the old path, which gathered/scattered reserved slot 0); VLLM_T4_GDN_RECURRENT_PREFILL_MAX default 64 vs the longest
              prefill row. Serve args = T3-007's. Measured on workspace t4-003m (ws create --from T4-003; diff identical).
  results:    Observed: GPU test (workspaces/t4-rp2/test_rp2.py): N=3 sequences (lengths 1/17/64), strided indices, fresh/resumed mix,
              one idx=0 row: outputs within 1.2e-4..4.9e-4 abs (~4e-3 rel) of the old path, states ~3e-3 rel; all other slots bitwise
              unchanged. Harness start 1: 851.4 tok/s; the path fired (20 log lines across both workers, every 500 calls).
  conclusion: Interpretation: no gain visible at one start; the expected ~+0.6% is below one start's resolution and the first start
              was already below the champion's mean, so this measurement cannot promote. Not pursued further (GPU budget, end of run).
validity:     leader-4: same function up to fp reordering (token-by-token form of the same recurrence, fp32; closer to an fp64 reference
              than the old chunk path, T4-001). Generic structural threshold, not fitted to the benchmark. GSM8K-250 would be needed for
              promotion (math path changes); GSM8K-50 88% on T4-001. Code review reviews/T4-4: no blocking issue for valid inputs;
              stride fix and hook branches correct (mock-executed); idx <= 0 semantics intentionally differ from the old path (old could
              read/overwrite reserved slot 0); "no parallelism advantage" in the code comment overstates: the 64 rule avoids the multi-
              kernel chunk pipeline for single-chunk inputs (fair, generic). GPU coverage of resumed/prefix handoffs is indirect.
