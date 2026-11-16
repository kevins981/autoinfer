team:         T5
builds_on:    T5-001
performance:  "within noise / not a candidate: 889.9 tok/s (starts 896.6 / 883.2) vs the champion T5-001 895.4; start 2 below (eval/1, GPU job 120)"
summary:
  hypothesis: T3-001's side-stream L2 prefetch of the next GEMM's weights during each decode hmar all-reduce was capped at 16 MiB because
              >=32 MiB lost (T3, job 47) while ~half of the ~77 MB per GDN layer of state writes sat dirty in the 72 MB L2. T5-001 cut
              those writes by ~2/3 (job 118), so a larger prefetch might now fit. Size chosen by hardware reasoning, not by benchmark
              runs (manager ruling): the ~47 us AR window can stream ~40 MB at ~0.85 TB/s; 24 MiB is the next step below T3's losing 32.
              Napkin < +1%.
  change:     T5-001 + one code default: vllm/distributed/device_communicators/l2_prefetch.py VLLM_T3_L2_PREFETCH_MB "16" -> "24"
              (workspace t5-pf24; change.diff = ws diff t5-pf24, measured). Serve args = T5-001's.
  results:    Observed: eval/1 (job 120, STOP_BELOW=895.4, no GSM8K: reads only): 896.6 / 883.2 tok/s (mean 889.9), acceptance 2.33,
              TPOT p50 16.32/16.31 ms (T5-001: 16.22/16.31), TTFT p99 998/1052, TPOT p99 32.0/30.3, latency gate pass, 128K pass.
  conclusion: Interpretation: no gain demonstrated in this run; start 1 within noise of T5-001, start 2 -1.4%. Mean -0.6%, within the run's
              job-to-job drift, so a weak negative (a screen, not a measured loss). 16 MiB stays.
validity:     leader-5: reads only (prefetch kernel loads into L2 and discards), no math change; size chosen by hardware reasoning
              (stated above), not selected on benchmark data. Not a candidate.
              Review: reviews/T5-3 (numbers confirmed; wording adopted; the docstring in l2_prefetch.py still says default 16: cosmetic, not a candidate so not fixed).
