# T4 write-up (leader-4, sweep team; closing)

**Wins: none.** Records: T4-001 (screen negative), T4-002 (screen negative), T4-003 (stopped after 1 start, -0.26%), T4-004 (finer prefill graph grid: OOM at startup).
GPU: ~30.8 of 45 GPU-min used (profile 12, recurrent prefill screens ~9, fusion ~11 incl. a timed-out fire check, harness 3.2 + 1.6).
Map: reports/T4-sweep.md, T4-sweep/1-6, update-1, update-2. Reviews: T4-1 (map), T4-2, T4-3, T4-4, T4-5 (closing decision).

## What I learned
1. **nsys misled my calibration of mixed steps.** The node trace showed ~60-row mixed steps at 72-122 ms with ~1730 eager launches
   and thousands of CPU-launch gaps, so I ranked "mixed steps are launch-bound" first. Unprofiled, the same steps cost 56-66 ms.
   task/PROFILING.md warns about this; I applied the warning to decode but not to mixed steps. The fix is cheap:
   `--enable-logging-iteration-details` gives per-step elapsed time without a profiler. Calibrate step costs with it, and use nsys
   only for the GPU-side picture.
2. **Fusion inside CUDA graphs buys little.** A 2.8 us norm kernel after a ~44 us host-mapped all-reduce costs about its own time in a
   graph; fusing it into the all-reduce saved nothing measurable (T4-002).
3. **The dual-form rewrite (framework 03 §4.2) is a length-dependent trade on this GPU.** The token-by-token GDN recurrence costs
   ~0.7 us/token/layer of GPU time against ~390 us of CPU per call for the chunked pipeline. It wins only for short prompts, by a
   margin too small to resolve with the harness's noise (T4-001, T4-003).
4. **The framework found the right structural sites, but the time left only allowed the cheap ones.** In GDN state Keep/Recompute (02 §4.3/4.4), the
   speculative verify writes 3 fp32 states per request per layer, and those slots are ~3.6 GB of the 4.4 GiB KV budget. That is
   both a bandwidth and a capacity lever (it gates k=3). It needs the slot contract changed in several readers. T5 took it.

## For the next team
- Price mixed steps from unprofiled per-iteration timing (bank/T4-001 eval/iter_quickbench/table.md has bins by prompt length).
- DeltaLog with an external log (frees spec slots) -> then k=3.
- The recurrent prefill kernel (workspaces/t4-rp2) is correct per review T4-2's fixes. It could be part of a future stack if someone
  has starts to spare.
