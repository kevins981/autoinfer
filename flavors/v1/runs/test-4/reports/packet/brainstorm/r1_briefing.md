# Round 1 briefing (run test-4)

## The problem
Serve Qwen3.8-27B-FP8 faster on 2x RTX 4090 (sm_89, 24 GB each, no GPU peer-to-peer: PCIe through
the host). Metric: output tok/s at concurrency 16 on the official ShareGPT benchmark (400 prompts,
~260 input / ~228 output tokens mean, temperature 0). Full task: task/TASK.md. What is allowed and
forbidden in this run: task_definition.md (read it: e.g. activations stay bf16, comms stay at
baseline precision, spec decode must be exact; drafter internals may change). Focus: focus.md.

## Facts
- Baseline server: task/harness/serve_args.sh (TP=2, max context 131072, 16 seqs, FP8 KV 5 GiB,
  MTP spec decode with 2 draft tokens, Marlin weight-only FP8 with bf16 compute).
- BASE official (bank/BASE/official/1/results.json): 635.9 tok/s (trials 635.9 / 635.9);
  TTFT p50/p99 235/1341 ms; TPOT p50/p99 23.0/46.1 ms; MTP acceptance length 2.345 (per position
  0.78 / 0.56); GSM8K 94.4%; 128K request passes.
- Gates: TTFT p99 <= 1609 ms, TPOT p99 <= 55.3 ms, GSM8K-250 >= 89.6%, one 128K request served.
- Noise: ~0.5% CV within a server; up to ~1.6% across jobs (TASK.md).
- Nothing has been tried yet in this run: BANK holds only BASE; tree.md is empty.

## Where to look (starting points, not a reading list)
- Raw baseline profiles: task/baseline/ (nsys_c16/: per-kernel nsys SQLite + summaries;
  ncu_marlin/: ncu of the FP8 GEMMs). How to query them: task/PROFILING.md. Nobody has analyzed them.
- Engine code: $CACHE_ROOT/tasks/qwen38-27b-fp8-2x4090/baseline/vllm (after `source task/env.sh`).
- Model config: huggingface_hub.hf_hub_download("Qwen/Qwen3.8-27B-FP8", "config.json").
- New profiles if you need them: see your instructions (gpu queue, unbudgeted, keep jobs short).

## What I need
3-6 ideas for the next round: 2 teams, each with ~55 GPU-minutes and one worker. That's enough for
a few unofficial jobs plus ~2 official evals per team. Write them to brainstorm/r1/<persona>.md
and message manager.
