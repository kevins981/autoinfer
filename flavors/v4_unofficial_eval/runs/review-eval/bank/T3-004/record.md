team:         T3
builds_on:    T3-003 (T3-002's greedy fast path, made exact)
performance:  "within noise, not a candidate: 827.6 tok/s (starts 817.0 / 838.3), +0.2% vs the champion T3-003 (826.1); start 1 below (eval/1, GPU job 69)"
summary:
  hypothesis: T3-002's greedy target fast path (no 6 MB/rank logits all-to-all) with the review T3-3 gaps closed stacks on the fp8 head.
  change:     by worker leader-3.w1 (workspace t3-g2): T3-002 + NaN handled by exact fallback (per-row NaN flag from the argmax kernel rides
              the [rows,3] all-gather; any NaN on any rank -> old path for that step; costs one host sync per step), lora_config must be
              None, num_new_sampled_tokens_per_step must be 1. serve_args = T3-003's.
  results:    Observed: harness (job 69) 827.6 (817.0, 838.3), acceptance 2.330, TPOT p50 17.92/17.62 ms (T3-003: 18.05/17.78),
              TTFT p99 948/1025, TPOT p99 31.6/40.3, gate_latency pass, 128K pass; fast path active in both starts.
              Unit test agents/leader-3.w1/greedy_unit2.log (job 68): 0 mismatches on 393 non-NaN requests vs rejection_sample (ties, inf),
              78/78 NaN batches flagged. c=1 text identical.
  conclusion: Interpretation: on top of the fp8 head the fast path no longer resolves a gain (TPOT p50 -0.1..0.15 ms); the per-step host sync
              (fallback decision) likely eats part of it. A sync-free exact variant would need the NaN decision on the device. Dropped.
validity:     Exact (review T3-3 turn 2: both P2s resolved, no remaining correctness blocker; fallback symmetric across ranks; the host sync
              affects overlap only). Not a candidate.
