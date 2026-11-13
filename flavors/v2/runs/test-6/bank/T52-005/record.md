team:         T52
builds_on:    T50-002 + T30-004/005 (layer-pipelined waves, ported) + T32-003 (captured GDN prefill core T32_GDN_CAP, ported) + T52-004 (link priority)
performance:  "official #1 1212.4 tok/s (+2.6% vs T50-002 1181.3); unofficial: same-job ABBA job 477 (400 prompts): all B > all A, +2.82% (B 1213.0/1220.0 vs A 1182.6/1183.8)"
summary:
  hypothesis: mixed-step all-reduces (copy-engine DMA lanes, token-split 2 chunks) are exposed because the two lanes interleave on the single
              D2H copy engine and because each layer joins both chunks (T52-004). Fixing link order (chunk 1's ARs wait for chunk 0's) and
              letting the two chunks run as layer-pipelined waves (no per-layer join; T30's code) hides much of it; capturing the GDN prefill
              core (T32) keeps wave steps from being host-bound.
  change:     (all Python; compiled files byte-identical to T50-002) env flags, ON by default, each "0" = off:
              - VLLM_T52_LINKPRI=1 (models/qwen3_5/nvidia/model.py, _qwen3_5_split_chain): events only: chunk 1 starts after chunk 0's out GEMM,
                chunk 1's AR1/AR2 wait for chunk 0's. Applies to joined split steps (>=256 rows below the wave threshold).
              - T52_WAVE=1, T30_WAVE_MIN=768 (vllm/t30_wave.py, model.py wave loop, model_runner.py split + eager, mamba_hybrid.py per-wave metadata,
                gdn_attn.py): steps with >=768 scheduled tokens run as 2 waves (64-aligned split inside a request, all spec/decode rows in wave 0);
                GDN state handed over through the cache (wave 0 writes the cut request's state, wave 1 continues from it); attention reads the
                KV cache as in the joined path. T52_WAVE_PRI=1: FIFO link order across waves.
              - T32_GDN_CAP=1, T32_CAP_NTMAX=16 (mamba/gdn/t32_cap.py, qwen_gdn_linear_attn.py, gpu_worker.py): GDN chunked-prefill core captured per
                (N<=3 seqs, NT<=16 chunks) key; per-wave metadata copied into the static buffers before each replay; core(A,L+1) waits core(B,L).
              Test-only flags off: T52_WAVE_CTL, T29_HOSTPROBE, T30_WAVE_BENCH/CHECK, T32_CHECK*.
              Work by leader-52.w1 (agents/leader-52.w1/NOTES.md, port/).
  results:    Observed: ABBA job 477 (same workspace, A = flags off = T50-002 behaviour, B = flags on via env; 400 prompts, fresh starts):
              A 1182.6/1183.8, B 1213.0/1220.0 tok/s; acceptance 2.655/2.663 vs 2.660/2.673; p99 TTFT 834/662 vs 523/595 ms; p99 TPOT 25.6/24.1 vs
              21.1/22.3 ms. GSM8K-250 on the last B server 93.6%. 128K long request (B) pass. Chain microbench basis: T52-004 (WP -12..-15% layer).
              Host probe (job 475): wave steps host 120 vs GPU span 131 ms with captured cores (eager cores 196 vs 178); below ~770 rows host-bound
              -> MIN 768. Capture: 2160 graphs, free GPU memory after capture 1.00 -> 0.45 GiB. Correctness (job 474/475): layer-0 stages vs joined
              rows: qkvz bitwise, ba/core rel 2.2-2.6e-3, AR out 1.3e-3, MLP ~1e-2 (M-dependent GEMM rounding); captured core bitwise vs eager;
              c=1 greedy: on/on and off/off identical, on/off diverge at tokens 9-54 (fp reordering).
              Defaults flipped after the ABBA (B ran with env); clean-env import check shows the same resolved values (no server run on defaults).
  conclusion: Interpretation: +2.8% same-job from scheduling the mixed-step copy-engine ARs (link order + no per-layer join) with captured cores.
              Remaining in the direction: rank skew (peer D2H lag ~100 us in the chain), the head/tail of each wave pipeline, eager non-core pieces
              of wave steps (host ~120 ms), steps < 768 rows.
validity:     leader-52 review against task_definition.md: same math up to fp reordering: waves split the rows of a step (row-independent GEMMs
              run at a different M -> different tile/accumulation order only); the GDN chunked-prefill split is at 64-token aligned positions inside
              a request, the recurrent state handed over in its fp32 cache dtype; attention reads the FP8 KV cache exactly as the joined path does;
              captured core is bitwise vs eager; link priority is stream ordering only. No change to precision, quantization scheme, comm precision,
              sampling or spec-decode verification; FP8 activation scheme untouched (no r-deferral used). Thresholds are on step size only, nothing
              benchmark-specific. Memory: 128K gate passed with the capture on. GSM8K 93.6% (= champion official 93.6%).

official #1 (2026-11-11T23:44): 1212.4 tok/s at c=16 (mean of 2 trials: 1210.1 / 1214.8, CV 0.3%) (+90.7% vs BASE), GSM8K 93.2%, MTP acceptance 2.66, TTFT p50 145 ms, TPOT p50 11.8 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
