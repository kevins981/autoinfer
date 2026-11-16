team:         T1
builds_on:    T1-008 (flags only relative to T1-008; its code incl. robustness fixes)
performance:  stopped after 1 start, 764.2 tok/s (-0.9% vs the champion T1-007's mean 771.3; T1-007's starts were 762.6 / 779.9).
              A screen negative, not a measured loss.
summary:
  hypothesis: T1-004 showed k=3 raises acceptance but was capped at 12 running requests by a 4.25 GiB KV budget; with
              ~5.5 GiB all 16 fit, so k=3 might pay on top of the faster champion.
  change:     flags only: T2-004/T1-007 serve args with num_speculative_tokens 3 and --kv-cache-memory 5905580032 (5.5 GiB).
  results:    Observed (trial 1, GPU job 43): 764.2 tok/s; acceptance 2.698 (vs 2.340 at k=2, T1-007); mean TPOT 19.41 ms vs
              19.64 (T1-007 trial 1), mean TTFT 246 vs 230 ms, duration 118.8 vs 119.0 s; KV capacity 286,466 tokens, mostly
              "Running: 16, Waiting: 0, KV usage 98.6%" (occasionally 15 running + 1 waiting); 0 preemptions; no OOM;
              gate_latency pass; 128K not reached (stopped).
  conclusion: Interpretation (revised after review T1-5): one start at 764.2 tok/s is consistent with roughly neutral
              performance vs T1-007 (+0.2% vs its start 1, -2.0% vs its start 2), without establishing a gain or loss.
              Acceptance length +15.3% (2.698 vs 2.340; draft acceptance rate ~57% vs ~67%) coincided with higher ITL
              (median 40.8 vs 35.0 ms, +16.5%), consistent with costlier steps offsetting the benefit (draft vs verify vs
              comm cost not isolated). The T1-004 queueing bottleneck is relieved (server queue time 0.032 s vs 1.32 s);
              5.5 GiB held 16 running requests in this benchmark with little headroom. Deprioritized (budget decision).
validity:     flags only; exact speculative decoding. A KV change needs the 128K test before any promotion (not reached).
              Reviewed: reviews/T1-5 (codex); corrections adopted.
