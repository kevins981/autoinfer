# leader-2.w20 R11: stack T1-028 + T1-032 + every-16 grid (128-512) on T1-030 (ws leader-2.stack11)
Port: T1-028's 3 files (fused_gdn_decode_kernel.cu, gdn/replay.py, qwen_gdn_linear_attn.py) were untouched by T1-030/T1-032
(per-file hunks identical in bank diffs), so copied from ws leader-2.mixleft; ws diff hunks for them == bank/T1-028 hunks; all
other T1-032 hunks unchanged. ws build ok (out/build.log); JIT host/dma AR exts pre-built from a shell (stack11 + arcomp).
Unit tests: job 133 (0.5 min): unit_test_layers 0/900 (neg 28), unit_test_mixed 0 (~44k checks; neg 6860) = T1-028's numbers.
serve_args.txt: T1-030's + 144,176,...,496 (87 sizes).
Measurement: abba_job.sh <root> <order> [S]; guards: no bench client before/after each arm, server healthy after bench,
bench.json required, server.log summary (graph mem, KV tokens, Running:16 count, comp on, NUMA bind).
Job 134 (S + ABBA): S graph 2.59 GiB (A 2.41), KV 267,721, 128K pass, gsm8k50 88%. 1A 849.4, 2B 849.7, 3B 853.4, 4A 857.1 (B -0.2%).
Job 135 (BAAB): 1B 846.9, 2A 848.2, 3A 848.4, 4B 850.9 (B +0.07%). All arms guarded OK.
Pooled 4+4: A 850.8, B 850.2 (-0.07%): no measurable gain; est. +0.5-1% not seen. TPOT/ITL p50 flat, acceptance same.
