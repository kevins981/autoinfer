team:         T7
builds_on:    T5-001 (flags only: its code unchanged)
performance:  stopped after 1 start, 888.9 tok/s (-0.7% vs the champion T5-001's mean 895.4; its starts were 898.7 / 892.1).
              A screen negative (roughly neutral), not a measured loss.
summary:
  hypothesis: k=3 MTP drafts had last been tested on older code (T1-009 on T1-007: one start, ~neutral) with a bf16 draft head
              (1.48 ms/draft step at M=16) and the 3-write GDN kernel. On the champion the draft step is ~0.78 ms (FP8 draft head) and
              GDN rows are cheaper (DeltaLog), so the 3rd draft should now pay: napkin +2-4% (reports/T7-sweep/6-ranking.md #1).
              Capacity: GDN spec-state slots count against the KV budget; k=3 needs ~5.5 GiB for 16 running requests (T1-004/T1-009).
  change:     flags only: T5-001 serve args with num_speculative_tokens 3 and --kv-cache-memory 5905580032 (5.5 GiB, as T1-009).
              Workspace t7-ns3 = ws create --from T5-001 (change.diff identical to T5-001's). DeltaLog handles T=4 (kMaxLogRows=3),
              HMAR <= 96 rows (64), GEMM configs include M=64. (An earlier submission at 4.4 GiB, job 122, was cancelled before running.)
  results:    Observed (GPU job 123, STOP_BELOW=895, trial 1 only): 888.86 tok/s; acceptance length 2.70 (per position 76.7/54.1/39.0%)
              vs 2.35 (78.2/56.7%) for T5-001 trial 1; median ITL 32.81 ms vs 28.74 (+14.2%); mean TPOT 16.68 vs 16.74; median TPOT
              16.43 vs 16.22; TTFT p99 1045 ms, TPOT p99 35.2 ms (gate pass); duration 102.0 vs 101.3 s. Server: mostly 16 running,
              0 waiting, KV usage 98.6% (a few samples 15 running / 92.4%, one with 1 waiting); graphs 1.86 GiB, no OOM.
              128K not reached (stopped).
  conclusion: Interpretation: the +15% tokens per step is eaten by a +14% step time (ITL), ~+4 ms per step vs the napkin's +2.6-3.1 ms;
              same picture as T1-009 on older code, so the cheaper draft head did not change the balance. The extra cost is not
              isolated (no profile): candidates are the 64-row verify (AR, attention, GDN), the 3rd draft step, and the KV running at
              98.6% (occasional 15 running). k=3 likely needs both freed spec-state slots (T5 writeup #2) and a cheaper per-row step.
validity:     flags only; exact speculative decoding (method unchanged, one more draft token, verification against the target as
              before). KV change would need the 128K test before any promotion (not reached). No benchmark keying.
              Review: the review tool returned "backend error" twice (07:04-07:05 UTC); not reviewed.
