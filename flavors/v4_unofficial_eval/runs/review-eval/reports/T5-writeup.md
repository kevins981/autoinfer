# T5 write-up (leader-5, sweep team; worker leader-5.w1)

## Result
- **T5-001 GDN DeltaLog: champion, 895.4 tok/s** (+4.9% vs T3-007's 853.6, +40.8% vs BASE). Bitwise-exact change to the GDN
  MTP decode kernel: store the committed state after verify row 0 plus a small log of rows 1..T-1 (normalized k, delta, decay)
  instead of 3 full fp32 states per (request, v-head, layer); replay the accepted rows next step; materialize the old format
  before every other consumer; off within 64 tokens of 1600-token prefix-cache block boundaries. bank/T5-001, job 115.
- T5-002 (L2 prefetch 16 -> 24 MiB on T5-001): weak negative, 889.9 (896.6 / 883.2), job 120.
- GPU: ~54 of 90 GPU-min used (profiles 77/78/118, ncu 85, worker tests 90-111, harness 115, 120).

## What I learned
1. **Price written bytes where they are evicted, not where they are written.** ncu (job 85): the old kernel put ~77 MB per layer
   into L2 but only ~38 MB reached DRAM during the kernel. My napkin (+2.5-3.5%) counted only the kernel. The profile (job 118)
   showed the following GEMMs and all-reduces also ~0.8 ms per decode step faster: the deferred write-back had been taxing them.
   Measured +4.9%. On a 72 MB-L2 GPU, the framework's per-window byte count needs a write-back term.
2. **Calibrate with the engine's own timer, not nsys, for eager work.** My first calibration read mixed steps as host-bound
   (~110 ms floor). That was nsys inflation, and a Prometheus average matched it only by coincidence. T4-001 had already shown
   it (`--enable-logging-iteration-details`). Compare distributions, not means.
3. **Engine paging dimensions are graph indices too.** The DeltaLog bug: the GDN layers are spread over 4 KV-cache groups, and
   a new state-index field was not regathered per group. Only c=16 with staggered arrivals (mixed steps) exposed it:
   GSM8K-50 30%. A CPU code review with the real update_block_table() found it in one turn.
4. **Bitwise exactness is a debugging tool.** Because the change is bitwise exact, every check was binary: unit test, c=1 text,
   in-graph old-op reference. The worker could bisect the c=16 failure to the integration, not the math, in 3 GPU jobs.
5. **Splitting work across teams worked.** With T4: T5 took DeltaLog, T4 took the AR+norm fusion and k=3. With T6: T6 took
   my #2 (mixed-step GEMM->AR overlap). There was no duplicate GPU spend.

## What's left (ranked)
1. DeltaLog for mixed-step spec rows (route them through the DL CUDA kernel): removes the 1.5 ms per mixed-step materialize and
   2 of 3 state writes there. ~+0.5-1%.
2. Log outside the spec slots + allocator change (num_speculative_blocks): frees ~2.4 GB/GPU of GDN state slots. That unblocks
   k=3 drafts (T4's #3, +2-3%). The largest structural follow-up.
3. P2 hygiene: allocate the fmt flags with the KV-cache binding. Today the champion config is safe because --kv-cache-memory
   skips the profiling rebind. Also strip the env-guarded debug code.
4. vLLM already contains delta-rule replay machinery (Kimi KDA "RecoverSSM", Mamba2 "ReplaySSM"). Porting the full version to
   Qwen GDN would subsume 1-2.

Map: reports/T5-sweep.md (files 1-6, update-1, update-2). Reviews: T5-1 (map), T5-2 (code, 3 turns).
