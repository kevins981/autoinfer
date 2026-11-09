# Measurement cost: making each experiment cheaper in GPU time

Question (human, 2026-11-09): the GPU queue is the bottleneck of a run's wall clock. How do we
make each experiment's measurement much cheaper while keeping most of its accuracy, so a fixed
budget buys more decisive experiments? Analysis of v2 test-6, discussion with the monitoring
session that watched test-6, and the decisions. Task-level where possible (all flavors).

## Where test-6's GPU time went (498 GPU-min, 189 jobs, 16.5 h)
- **Server startup: 46% of all GPU time** (144 starts, mean 95 s). In the same-job A/B jobs that
  decide whether a change works, startup was 55% of the job. Official evals: 3 jobs, 6%.
  Microbenchmarks (no server): 130 jobs, 24%, median under a minute.
- Failed or timed-out jobs: 56 min (11%), mostly servers dying at startup and one 14-min A/B
  (job 119) killed because its first start compiled cold.
- Cold starts (160-210 s) vs warm (60-80 s): the difference is JIT compiles of the record's own
  extensions (host_staged_ar 38 s, dma_staged_ar 40 s, lmh_codec 49 s). Warm start = process
  spawn/imports ~25 s, model load ~10 s, CUDA-graph capture ~20 s, init ~5 s.
- The GPU was busy only 50% of the wall clock (median queue wait 0, p90 4.5 min): with two teams
  the binding constraint was per-team latency (a worker waits 10-15 min per A/B), not GPU
  throughput. Throughput binds as soon as more teams run; both improve with the same fixes.
- Noise, not bench length, forces the repeats: wins were +0.7%, +0.8%, +1.1%; same-server repeat
  benches vary ~0.5%, fresh starts of the same code ~1% (KB36: sometimes 2.5%). Logged GPU clocks
  do not explain it (GPU0 pinned at 2235 MHz; GPU1 +-40 MHz, uncorrelated with throughput).
  42 official trials, pairs of the same code in one job: median |delta| 0.3%, one 2.6% outlier.
- A typical decisive A/B: 4 fresh starts x (70 s start + 100-140 s bench of 300-400 prompts)
  = 10-12 GPU-min.

## Decisions (2026-11-09)
1. **JIT extension compiles no longer repeat per workspace** (done, task level). Cause: torch
   builds a JIT extension with the source path relative to its build folder; a new workspace
   (every `official_eval` creates one, workers create many) means a new path and a full
   recompile even for byte-identical sources, inside the GPU job with both ranks waiting. Fix:
   `tasks/<task>/nvcc_jit.sh`, set as `PYTORCH_NVCC` by the task's `env.sh`: copies each source to
   a content-addressed path (`$RUN_CACHE/jit_src/<sha1>/`) and compiles through the shared ccache.
   CPU test: 41.7 s miss, 0.4 s hit from another path, an edit misses, the identical edit elsewhere
   hits. GPU test (test-6-smoke job 3, champion T27-002's code from two brand-new workspace
   paths): start 1 ready after 229 s (compiles: 44 + 39 + 58 s), start 2 after 74 s (the same
   phases took 1 + 1 + 12 s; the 12 s is lmh_codec's real encode check, not a compile). Saving: ~2 min per official eval, ~40 s on the first start of most worker jobs
   (~10% of test-6's GPU minutes), and no more job-119 timeouts. Sources with local includes are
   passed through unchanged (not cached across paths). Triton and torch.compile caches were
   already content-keyed per run.
2. **Early stopping on a clear loss** (done, task level + v2 prompts). After the first A,B pair,
   stop if B trails A by more than 1.5% (~2.5 sigma of a pair difference). Losses were the
   majority of test-6's A/Bs, so this roughly halves their cost. Only losses stop early; a win
   needs every planned start (all B above all A) and the official eval. A stopped run is recorded
   as *screen negative (1 pair)*, not a measured negative: T7's "prefill AR hiding doesn't pay"
   deterred work until T9 showed it was a NUMA artifact. `tasks/<task>/scripts/abba.sh` runs the
   protocol (fresh interleaved starts, same prompts, early stop, verdict line, acceptance per
   start); TASK.md states the rule and the measured noise behind the threshold.
3. **Runtime-toggle A/B on one server: rejected as a requirement.** The idea (one server holding
   A and B behind a flag, cache reset between steady-state windows) would cut an A/B to ~4.5 min,
   but most of what we optimize runs inside CUDA graphs: a Python flag is read at capture time,
   and making graph-captured kernels switchable costs either +1.1 GiB (two graph sets; the
   champion uses 23.8/24 GB) or extra branches the size of our wins. As a requirement it would
   steer agents toward small switchable items. It may return as an *optional* tier for
   scheduler/Python-side changes, only after an A/A/A/A calibration and reproducing test-6's known
   results (T9-005 +0.7, T14-004 +0.9, T27-002 +0.76, T22-002 -0.9, T21 -4.5). Also: a
   steady-state window misses the drain (the last ~2% of a trial with <16 in flight), where
   adaptive-k and deeper-speculation items act; one server samples one memory placement.
4. **Next: a step-replay tier for kernel/communication changes** (to build). Record one official
   trial's schedule (per step: decode rows, prefill chunk per request, context lengths, accepted
   draft tokens, token ids for teacher forcing) and replay the *full* model step on both ranks
   (all layers + drafter, real host-memory AR path, CUDA graphs on, same memory layout) without
   scheduler, HTTP or sampling, timing the critical path per rank. The GPU never idles in this
   workload (KB46), so the sum of step times predicts the trial time for changes that don't alter
   the schedule. Noise ~0.1-0.2% (T14-002's nsys critical-path A/B showed -0.59 s/trial with a CI
   excluding 0 where end-to-end tok/s could not). Cost: two loads + captures (~70 s each) plus
   ~1 min of replay per variant: ~3-4 GPU-min per comparison vs 10-12. Invalid, and must say so,
   for anything touching the schedule, Python per-request paths or acceptance (T27-003's -8.5%
   plumbing penalty lived outside the GPU step). Must replay the real step, not isolated kernels:
   test-6's microbenchmarks disagreed with the server repeatedly (T2 3x interference, T8, T9,
   T22/KB78 L2 coupling). Calibration before agents rely on it: A/A spread, then T14-002,
   T27-002 (~0.44 ms/decode step), T22-002 (-0.9%), T16.
5. Smaller items, not done: shortest-job-first with aging in the GPU queue (micro jobs waited
   behind 10-14 min A/Bs; costs no GPU minutes); a CPU-side pre-flight for servers that die at
   startup; official evals stay as they are (6% of spend, the trust anchor).

## Tiers, as they stand
| tier | cost (GPU-min) | noise floor | valid for |
|---|---|---|---|
| napkin / roofline | 0 | - | everything |
| kernel microbench (1 GPU) | <1 | kernel-level, often disagrees with the server | screening |
| step replay (`vllm bench forward`, evaluation paused) | ~3-4 per A/B | ~0.2% (to measure) | kernel/comm changes with an unchanged schedule |
| fresh-start A/B, early stop (`abba.sh`) | ~5 (loss) / ~9 (win) | ~0.5% per pair | anything |
| official eval | ~9 (after item 1) | - | promotion |

Promotion stays on fresh-start A/B plus the official eval throughout.
