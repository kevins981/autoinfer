# Task: Qwen3.8-27B-FP8 on 2x RTX 4090 (parallelization is open; the baseline uses TP=2)

Decided (human, 2026-11-06): the baseline serves with **FP8 KV cache** (`--kv-cache-dtype fp8
--kv-cache-memory 5368709120`, see `harness/serve_args.sh`): official BASE = 630.6 tok/s at c=16,
GSM8K 92.8% (vs 656 tok/s with bf16 KV in qwen38_opt; FP8 KV fits ~2x the 128K-context requests).

TODO: objective, hardware, hard constraints (FP8 weights, FP8 KV cache, ...), allowed/forbidden changes, correctness criteria (what a validity review checks).

TODO: include the baseline profiling results (nsys kernel breakdown, ncu on the hot GEMMs; from
`qwen38_opt`): every agent can read them, and round 1's brainstormers start from them.
