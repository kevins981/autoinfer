# Round 1 briefing (run test-3)

## The problem
Serve Qwen/Qwen3.8-27B-FP8 faster on 2x RTX 4090 (sm_89, 24 GB each, FP8 tensor cores, **no GPU
peer-to-peer**: PCIe through the host). Metric: output tok/s at concurrency 16 on a fixed
ShareGPT benchmark (400 prompts, input mean ~260 tokens, output mean ~228, temp 0,
/v1/chat/completions, closed loop). Full objective, gates and hardware: task/TASK.md.
What is allowed / forbidden in THIS run: task_definition.md (read it: e.g. activations must stay
bf16, so no W8A8/FP8 compute; exact spec decode only; one 128K request must still fit; no
quantized communication). Focus: focus.md (whole engine, everything open).

## Facts
- Baseline server: task/harness/serve_args.sh (TP=2, max-model-len 131072, max-num-seqs 16,
  FP8 KV with a 5 GiB budget, MTP spec decode with 2 draft tokens, Marlin weight-only FP8 GEMMs
  with bf16 compute, vLLM commit 4527515cd7).
- This run's BASE official numbers (bank/BASE/official/1/results.json): 631.4 tok/s, TTFT p50 251 /
  p99 1487 ms, TPOT p50 23.2 / p99 48.4 ms, GSM8K 92.8%, MTP acceptance length 2.34 (0.779, 0.56).
  The latency gates are relative to this run's BASE.
- Raw profiles of the baseline, not analyzed by anyone: task/baseline/ (nsys timeline of both
  GPUs at c=16 with CUDA graphs traced per kernel; ncu full metrics + source-level of the Marlin
  FP8 GEMMs). How to query them: task/PROFILING.md.

## What was tried
Nothing yet: this is round 1. BANK has only BASE.

## Where things are (starting points, not a full list)
- Engine code: /scratch/kevinsong/learning/autoinfer/.cache/tasks/qwen38-27b-fp8-2x4090/baseline/vllm
- Helper scripts for GPU jobs (profiling included): task/scripts/
- Workspaces and the GPU queue: `ws --help`, `gpu --help`.
