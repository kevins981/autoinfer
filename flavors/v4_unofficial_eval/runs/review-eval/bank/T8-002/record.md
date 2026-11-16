team:         T8
builds_on:    T5-001 (profile of the champion's code and serve args; no change)
performance:  n/a (profile only, GPU job 125, ~1.8 GPU-min)
summary:
  hypothesis: the champion's mixed (prefill+decode) steps carry a large non-GEMM excess (T8 6-ranking #1, from job 118's node trace).
  change:     none. nsys TRACE=graph (accurate graph-launch timing), 32 prompts at c=16, T5-001 code (workspace t8-champ, ws diff
              == T5-001's, md5 3a07bc41) + T5-001 serve_args (prof.sh). Output: agents/leader-8/prof_graph/ (steps.txt, steps.csv, sqlite).
  results:    Observed (GPU 0): 353 decode steps: period 22.9 ms, GPU idle in window 0.14 ms. 20 mixed steps (median 374 prefill + 45
              decode tokens): period 115 ms, busy 116, idle in window 23 ms. Per step (prefill tokens -> period / GPU busy ms):
              59 -> 114.5/50.2, 61 -> 113.1/49.9, 65 -> 110.1/49.9, 110 -> 115.7/58.5, 127 -> 115.2/60.9, 132 -> 110.3/85.2,
              358 -> 114.8/116.3, 389 -> 111.1/116.7, 429 -> 114.3/126.6, 482 -> 122.5/154.8, 740 -> 154.1/206.9, 849 -> 155.5/209.1.
              (busy > period where kernels overlap across streams/steps.)
  conclusion: Interpretation: a mixed step has a ~110 ms period FLOOR that is independent of prefill size up to ~450 tokens. With a
              60-130-token prefill the GPU is busy only ~50 ms of it (~60 ms idle), against a 22.9 ms decode step. So the excess is
              not GEMM compute and not prompt size. It is a fixed per-mixed-step cost: host-side work or a synchronization (candidates:
              prefill metadata / GDN prefill path setup, the DeltaLog materialize, eager pieces between graph pieces, rank skew).
              Profiling inflates host time, so absolute numbers are upper bounds. Napkin: the harness has ~400 mixed steps; removing
              ~40-60 ms of fixed cost from each saves ~16-24 s of ~101 s, i.e. +15-25% if it is real at the harness's speed. This is the
              largest gap found in T8's map. Next step: CPU-side sampling (py-spy on the engine core/workers during a mixed step), or
              the NVTX ranges in the same sqlite, to name the host cost.
validity:     n/a (measurement of the champion; nothing to promote).
