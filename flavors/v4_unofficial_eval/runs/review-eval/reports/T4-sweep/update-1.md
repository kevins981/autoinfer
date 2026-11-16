# T4 map update 1

## Tested since 6-ranking
| candidate | result | evidence |
|---|---|---|
| #1 mixed-step launch cost: one-kernel recurrent GDN prefill (03 §4.2) | **screen negative**: fires on every prefill step, GSM8K-50 88%, but -1.3% / -1.4% tok/s (1 start each, noise) | bank/T4-001; jobs 74, rp-iter |
| premise of #1 (mixed steps CPU-launch-bound) | **exaggerated by nsys** (review T4-2: the unprofiled CPU/GPU split is not resolved; elapsed_ms includes worker launch delays): per-iteration timing without nsys: mixed steps with <128 prompt rows 56-66 ms vs a ~52 ms GPU-work floor (nsys showed 72-122); 128-256: ~80, 256-512: ~120-130, >=512: ~200 ms; decode D=16 ~31 ms; mixed steps ~34% of time | bank/T4-001 eval/iter_quickbench/table.md |

## Verdicts changed
- 3-calibration B was calibrated on an nsys trace; nsys inflates eager launch cost (task/PROFILING.md says so; I applied the warning
  only to decode). Corrected mixed-step costs above (unprofiled, `--enable-logging-iteration-details`, a free per-step timer).
  The small-P "fixed excess" is far smaller than nsys suggested (56-66 vs 72-122 ms). Mixed steps carry substantial GPU work: GEMMs at M ~300-900
  (~210 TFLOP/s achieved) and 132 NCCL all-reduces.
- #2 (smaller token budget / chunked prefill into decode steps): with a small mixed step at ~56-66 ms vs decode ~31, a step carrying
  ~50 prompt rows costs ~+25-35 ms over decode; spreading prompts over more steps multiplies that: stays a loser. Dropped.
- New method row for the framework's calibration step: measure step costs with the engine's own per-iteration timer, not only nsys.

## New candidate
- Fuse the decode HMAR all-reduce with the following Gemma add-RMSNorm + dyn128 quant (one kernel per pair, 128 pairs per decode
  step): removes ~128 norm launches (~2.8 us each + launch gaps) and one pass over x: ~-0.4-0.6 ms per decode step -> ~+1%.
  Exact AR sum; the norm's sum-of-squares order changes (fp reordering) -> GSM8K. Building (worker task 5, on T3-007).

## Current ranking (vs champion T3-007, 853.6)
1. AR + norm + quant fusion (~+1%) — building.
2. GDN DeltaLog (~+1-2.6% decode) — not attempted: the slot contract (mixed Triton path, prefix-cache temporal copies) is too large
   for the time left. Note: a DeltaLog that keeps the log outside the spec slots would also free the 2 spec state slots per request
   (~2.4 GB/GPU at 16 requests: GDN states are ~3.6 GB of the 4.4 GiB KV budget), which is what k=3 needs. Biggest structural
   opportunity left in my map.
3. k=3 — blocked by GDN state capacity (needs a 4th slot, ~1.2 GB/GPU) unless 2. frees slots.
