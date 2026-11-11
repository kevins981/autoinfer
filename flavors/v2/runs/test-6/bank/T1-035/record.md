team:         T1
builds_on:    T1-030
performance:  "OFFICIAL #1 906.3 tok/s (+1.0% vs T1-030 897.3); unofficial: ABBA same job (job 140), B mean 861.9 vs A 852.9 tok/s = +1.05% (both B > both A); ITL p50 26.98 -> 26.48 ms"
summary:
  hypothesis: The GDN replay kernel stores the committed fp32 state S[0] every decode step. Keeping S[0] stale and letting the per-slot log hold up to 8 accepted-but-unapplied tokens skips about 2 of 3 state stores, about 0.6 ms/step (gate job 136: ITL 27.0 -> 26.36 with double replay modelled).
  change:     Lazy checkpoint in the GDN replay kernel (w21, ws leader-2.lazygdn.real):
              - flag byte = pending<<3 | last; the log holds up to log_cap=8 tokens (6.2 KB of the existing 64 KB per-head spec slot).
              - each step replays pending + accepted tokens. If S[0] is the source slot and replay+new <= cap, it skips the S[0] store and appends the new tokens after the replayed ones; otherwise it stores S[0] and restarts the log.
              - every flush (modes 0/1/2) replays pending tokens first; mode-3 clears reset them.
              - VLLM_GDN_LAZY_CAP (default 8; 0 = T1-030 behaviour), VLLM_GDN_LAZY=0 forces 0.
              No serve-arg change; memory, KV and graphs unchanged.
  results:    Observed:
              Unit tests (jobs 138/139), side by side with the stock kernel (outputs + stock-visible slots after flush):
              - cap 8: 4 seeds x 300 steps x 16 lanes, about 36k checks, 0 mismatches. cap 5 and cap 0: also 0.
              - About 65% of states had pending tokens; per seed about 190 re-prefills, 66-89 mode-3 clears with pending tokens, 76-92 align flushes with pending tokens.
              - Negative controls fail: 1152 / 5841 / 7134 / 2929 mismatches.
              Microbench: 65.4 -> 49.8 us/layer alone, 164.8 -> 150.7 with GEMM.
              ABBA (job 140, fresh server per arm, 200 prompts, c=16):
              - tok/s: 1A 851.6, 2B 859.2 (stats on), 3B 864.6, 4A 854.1
              - ITL p50: 26.99 / 26.50 / 26.47 / 26.98 ms
              - acceptance: 2.333-2.339 in every arm
              Greedy c=1, 10 prompts (5 cross the 1600 align boundary), 300 tokens each: 10/10 identical. gsm8k50 = 88% (= BASE subset).
              In-server flush counters [pre, post, spec, clear] = [4608, 9216, 0, 0]: align flushes with pending tokens do fire. Mode 3 with a live log is covered by the unit tests only.
  conclusion: Interpretation: about +1% real, exact by construction: the same fp32 ops in the same order; only when S[0] is written to memory changes. The step saves 0.5 ms vs 0.64 estimated; part of the store traffic was probably already absorbed by L2.
validity:     leader-2 reviewed the incremental diff vs T1-030 (kernel flag/log layout, log_base append, the flush kernel replaying pending tokens before last, cap check, wrapper env). The state stays fp32 end to end and the replay order is unchanged, so outputs are bitwise identical; the unit tests show this, with controls. No precision, KV, comm or serving change, nothing benchmark-specific. OK.

official #1 (2026-11-08T11:25): 906.3 tok/s at c=16 (mean of 2 trials: 903.6 / 908.9, CV 0.4%) (+42.5% vs BASE), GSM8K 92.8%, MTP acceptance 2.325, TTFT p50 174 ms, TPOT p50 16.1 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
