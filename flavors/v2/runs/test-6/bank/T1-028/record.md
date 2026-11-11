team:         T1
builds_on:    T1-026
performance:  "official #1: 880.0 tok/s (879.9/880.1) = T1-026 (880.9) within noise; same-job +0.2% (noise). Exact, harmless; not a new best"
summary:
  hypothesis: no-op GDN flush launches (modes 0/1 every step, mode 3 per layer in mixed steps) cost ~45 us/decode step and ~0.45 ms
              per mixed step; the mixed-step replay kernel (46 vs "25" us) may have headroom.
  change:     csrc gdn_replay_flush_kernel modes 0/1/3: one block per (req, head) looping over the layer pointer table (16x fewer blocks
              per no-op launch; per-layer work unchanged; mode 2 unchanged); mode-3 clear once per KV group per forward instead of per
              layer (replay.py _GroupClear; layers of a group share the metadata, hence the non-spec keys). Worker leader-2.w17,
              ws leader-2.mixleft (out/NOTES.md, unit tests, job_abba).
  results:    Observed: unit tests (jobs 103-104): batched multi-layer flush vs per-layer launches 0 mismatches / 900 layer checks
              (6 seeds; reference changed 1245 pairs; neg. control 28 mismatches); unit_test_mixed 0 / ~44k (neg. 6860).
              ABBA job 105 (200 prompts c=16): A 838.2, B 844.6, B 836.3, A 839.2; TPOT/acceptance same; gsm8k50 88%.
              Replay kernel like-for-like: 40.7 us decode @16 reqs vs 46.7 mixed @15 (~HBM roof; ~0.1% headroom): no change.
              Estimate 0.2-0.3%; measured +0.2% (noise).
  conclusion: Interpretation: exact and harmless; gain too small to resolve alone. Stack with the next small change.
validity:     leader-2 review: flush work per layer identical (same ops, same order per layer; only launch shape and dedup change);
              unit tests bitwise with working negative controls; no memory/graph change. OK.

official #1 (2026-11-08T08:32): 880.0 tok/s at c=16 (mean of 2 trials: 879.9 / 880.1, CV 0.0%) (+38.4% vs BASE), GSM8K 93.6%, MTP acceptance 2.322, TTFT p50 175 ms, TPOT p50 16.4 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
