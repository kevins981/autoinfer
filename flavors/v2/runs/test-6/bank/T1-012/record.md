team:         T1
performance:  "official #1: 749.7 tok/s (+2.6% vs T1-009 730.9, +18.7% vs BASE), GSM8K 93.6%, TTFT p99 1426, TPOT p99 43.3, accept 2.353, gates + 128K pass"
summary:
  hypothesis: the GDN MTP decode kernel stores the full fp32 state after each of the k+1 tokens (~75 MB writes/layer/step + spill
              into the next GEMMs); keeping one committed state + a log of last step's per-token inputs and replaying the accepted
              tokens next step halves state traffic (R3-B).
  change:     T1-009 + csrc gdn_decode_replay_kernel (load S[0], replay accepted tokens from the log in S[1] with the same fp32 ops
              in the same order, store committed S[0], run new tokens without state stores, log their inputs; flag byte per
              (block, value head) = logged count) + gdn_replay_flush_kernel (materializes the stock layout S[0..a-1] from S[0]+log
              before anything outside the fused kernel reads it: mode 0 before the align pre-copy, mode 1 before the post-copy,
              mode 2 at the top of the non-fused GDN path for spec requests, mode 3 clears flags of non-spec requests). New
              replay.py registry (flags sized max(8192, num_blocks) once at init, capacity asserted every flush). Dispatch in
              qwen_gdn_linear_attn.py; hooks in mamba_hybrid.py. Stock kernel and copy kernels unchanged. Default on;
              VLLM_GDN_REPLAY=0 opt-out; test-only VLLM_GDN_REPLAY_STATS=1 counters (default off). Falls back to stock unless V2
              runner, fp32 state, 1..3 spec tokens. Needs csrc build. Worker leader-1.w6, ws leader-1.gdnreplay.
  results:    Observed:
              - ceiling microbench (16 reqs x 3 tok, TP2 shapes, CUDA graph, per layer): stock 122.8 us alone / 267.4 with 2 GEMMs;
                last-store-only 61.8 / 213.7; replay R=2 65.6 / 217.7 (est. 50-60 us; in-server stock is 76 us -> L2 absorbs
                part of the writes, i.e. the spill).
              - unit test (out/unit_test.py, out/gdn_test_k.cu): stock vs replay side by side, 2 layers, random accepts, mixed
                token counts, joins/resets, random mid-stream flushes: outputs and stock-visible slots bitwise equal, 0
                mismatches over 6 seeds (N=6 x 100 steps, N=16 x 300 steps). Negative control (no flush): 106 mismatches.
              - in-server flush counters [pre, post, layer_spec, layer_clear]: align-crossing test (prompts 1454-1554 tok, block
                1600, 300 gen tokens) [1152, 1152, 0, 0], 12/12 generations identical ON vs OFF (c<=4); mixed-batch job (gsm8k50 +
                bench100) [1152, 0, 1611648, 0]. Mode 3 never fired live (unit-test only).
              - server same-job A/B (200 prompts c=16, only VLLM_GDN_REPLAY differs): 712.4 vs 697.3 tok/s; TPOT p50 19.57 vs
                20.41 ms; acceptance 2.359 vs 2.349. gsm8k50 88% (same as base on this subset).
  conclusion: Interpretation: ~2 ms/decode step saved (0.84 ms TPOT x 2.35), matching the ceiling estimate. v2 (fewer GDN spec
              blocks so k=3 fits at 16 reqs) is feasible but ~1-2 days (drop the stock per-token layout everywhere); not done.
validity:     leader-1 review: exact by construction (the same fp32 recurrence on the same inputs in the same order; the stock
              layout is rebuilt bitwise before any external reader), shown bitwise in the unit test with a working negative control,
              and the flush paths are shown to fire in-server. GSM8K/acceptance unchanged. Capability decided once at init; capacity
              overflow is an assert, not a silent mid-request switch. No change to KV accounting or graph sizes (flags: 8192 x 24 B
              x 48 layers ~ 9 MB/rank) -> not a memory-picture change requiring the 128K test, but official_eval runs the 128K
              request anyway. Residual risks: preemption/resume untested (re-prefill clears flags via mode 3 path); mode 3 only
              unit-tested. OK.

official #1 (2026-11-07T03:06): 749.7 tok/s at c=16 (mean of 2 trials: 746.0 / 753.5, CV 0.7%) (+18.7% vs BASE), GSM8K 93.6%, MTP acceptance 2.353, TTFT p50 206 ms, TPOT p50 19.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
