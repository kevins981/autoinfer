# Task: serve Qwen3.8-27B-FP8 faster on 2x RTX 4090

This is the task's **baseline definition**. At the start of each run the manager and the human
refine it into the run's `task_definition.md` (correctness, constraints, loopholes) and
`assumption_ledger.md`; where they differ, the run's files win.

## Objective and metric
Maximize **output tokens/s at concurrency 16** on the official benchmark, without changing what
the model computes.
- **Official benchmark** (`harness/eval.sh`, run only by `official_eval`): `vllm bench serve`,
  ShareGPT (400 prompts, input mean ~260 tokens incl. chat template, output = the reference reply
  length, mean ~228), `/v1/chat/completions`, temperature 0, seed 0, closed loop with 16
  concurrent requests. One trial (run-to-run noise ~0.5% CV; treat < 3% as noise).
- **Score**: output tok/s. **Baseline (record BASE): 630.6 tok/s**.
- **Latency guard**: a result is invalid if TTFT p99 or TPOT p99 is more than 20% worse than
  BASE's (BASE: TTFT p99 1355 ms, TPOT p99 41.2 ms). Other latencies are reported, not constrained.
- **Accuracy gate**: GSM8K-250 (fixed questions, temperature 0, thinking off) must be
  **>= 89.6%** (BASE 92.8%, standard error ±1.6; the gate is 2 SE below). A safety net: lossy
  changes are forbidden anyway (below).

## Hardware and software
2x RTX 4090 (sm_89, 24 GB each, FP8 tensor cores), **no GPU peer-to-peer** (PCIe through the
host), 48 CPU cores, 125 GB RAM, shared server. CUDA 13.0 toolkit (driver 580: no newer),
torch 2.14 + cu130, vLLM at commit `4527515cd7` (`workspace.json`). Details and the profile:
`baseline/PROFILE.md`.

## Baseline server
`harness/serve_args.sh`: TP=2, max context 131072, max 16 sequences, GPU memory utilization
0.92, FP8 KV cache (`--kv-cache-dtype fp8`, 5 GiB KV budget), MTP speculative decoding with 2
draft tokens, qwen3 reasoning parser, qwen3_coder tool parser, language model only. FP8
weights run through Marlin (weight-only FP8, bf16 compute; vLLM's default on sm_89).

## Fixed (hard constraints)
- The **model**: Qwen/Qwen3.8-27B-FP8, its weights and architecture.
- **FP8 weights** and an **FP8 KV cache**: nothing lower precision (no INT4/FP4, no pruning or
  distillation), no higher-precision copies that change the memory picture unfairly.
- **Sampling**: what the benchmark requests (temperature 0, max tokens, chat template, thinking
  as the request says) must be served as requested. Outputs must not be cut short or altered.
- The **benchmark, dataset, harness and accuracy gate** (agents never modify `tasks/`).
- The hardware: these two GPUs only.

## Open (everything else, within the fixed list)
Code anywhere in the engine (kernels, scheduler, memory, communication, compilation), the
parallelization scheme (TP/PP/other splits), serving flags (via a record's `serve_args.txt`):
batching limits, memory split, backends, CUDA graph settings, **speculative decoding** (method,
number of draft tokens), max context. The run's task definition may narrow this.

## Correctness (what a validity review checks)
- The served model computes the same function as the baseline: changes may reorder
  floating-point operations or change kernels, but not approximate (no lossy quantization,
  skipped layers, truncated vocabularies, early exits). Speculative decoding must remain exact
  (verification against the target model; at temperature 0 the output equals the target's).
- Nothing specific to the benchmark: no caching or precomputing outputs for its prompts, no
  tuning to seed 0 or to these 400 prompts, no detection of the benchmark.
- Every request is answered in full; the server stays a general OpenAI-compatible server (tool
  calls and reasoning parsing keep working).

## Files
- `harness/`: the frozen official eval (`eval.sh`, `serve_args.sh`, `gsm8k.py`, `summarize.py`,
  GSM8K data). Only `official_eval` runs it.
- `scripts/`: helpers for agents' GPU jobs (read-only, use as is or copy): `lib.sh`
  (`start_server`, `bench`, `gsm8k`, `stop_server`), `example_job.sh`, `profile_nsys.sh` (nsys +
  kernel-time breakdown), `kernel_breakdown.py`, `profile_ncu.sh` (ncu on chosen kernels of a
  live server), `ncu_hotlines.py` (top source lines by stall samples).
- `PROFILING.md`: what profiling data exists and how to query it (nsys SQLite, ncu Python API).
- `baseline/`: `PROFILE.md` (start here), `nsys_c16/` (nsys of BASE: report, kernel summary,
  breakdown), `ncu_marlin/` (ncu of the FP8 GEMMs: `marlin_c16.ncu-rep` full metrics,
  `marlin_c16_key.csv` the key numbers, `marlin_c16_src.ncu-rep` source-level with -lineinfo).
- `env.sh`, `workspace.json`: environment and workspace setup (used by `ws`).
- Profiling tools: `ncu` (Nsight Compute, in the env; works without root), `nsys` (Nsight
  Systems 2024.5, `/usr/local/cuda-12.6/bin/nsys`). Counters may only work on GPU 0
  (`--devices 0`).
