team:         T1
builds_on:    T1-030 + T1-028 + T1-032
performance:  "unofficial: null. 2 same-job ABBAs pooled A 850.8 vs B 850.2 (-0.07%); not evaluated"
summary:
  hypothesis: the exact small wins (T1-028 cheaper no-op flushes ~+0.2%, T1-032 Marlin single launch ~+0.4%, every-16 prefill graph grid
              over 128-512 ~+0.5%) add up to a measurable +0.5-1% on T1-030.
  change:     T1-030 + T1-028's 3 GDN files + T1-032's 3 Marlin files + serve_args with capture sizes every 16 between 128 and 512 (87
              sizes). Worker leader-2.w20, ws leader-2.stack11 (out/NOTES.md, out/abba_job.sh, job1/, job2/).
  results:    Observed: unit tests (job 133) as T1-028 (0 mismatches; neg. controls 28 / 6860). Stack server: graphs 2.59 GiB/rank (vs
              2.41), KV 267,721, Running 16, 128K pass, gsm8k50 88%. Job 134 ABBA: A 849.4/857.1, B 849.7/853.4 (-0.2%); job 135 BAAB:
              B 846.9/850.9, A 848.2/848.4 (+0.07%). Pooled A 850.8, B 850.2. TPOT/ITL/acceptance flat; TTFT p50 B ~same or higher.
              All arms guarded (healthy servers, no stray clients, node-0 slots, compression on). Estimate +0.5-1%; measured ~0.
  conclusion: Interpretation: these small exact changes don't add up to anything measurable on T1-030 (resolution ~±0.3% pooled);
              the every-16 grid costs +0.18 GiB/rank for no visible benefit. Don't adopt.
validity:     exact / fp-reorder only (as the component records). Not for eval.
