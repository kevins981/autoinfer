# TASK DEFINITION (run test-1; agreed with human; overrides TASK.md where different)
Metric: output tok/s at c=16 (official_eval). BASE 630.6. Noise <3%.
Gates: TTFT p99 and TPOT p99 not >20% worse than BASE (1355 ms / 41.2 ms); GSM8K-250 >= 89.6%.

## Correctness
- Mathematically lossless: same function as baseline up to floating-point reordering/kernel changes. No lossy quantization, approximation, skipped layers, truncated vocab, early exits.
- FP8 weights and FP8 KV cache; nothing lower; no higher-precision weight copies altering the memory picture.
- Reduced-precision communication (e.g. fp8 all-reduce) is NOT allowed (approximate) unless human later agrees.
- Speculative decoding (MTP, draft count) may change but must stay exact (verified against target at temp 0).
- Sampling/temperature/thinking/max tokens/chat template as requested; outputs never truncated or altered.
- Server stays a general OpenAI-compatible server (tool calls, reasoning parser work).

## Constraints
- Max context >= 131072 and one 128K request must still fit; KV capacity must not shrink below baseline's.
- --max-num-seqs MAY change (human OK: mathematically equivalent), but must run without GPU errors; other numeric parameters must also be validated to not crash/OOM.
- Allowed: engine code, kernels, serving flags (serve_args.txt), TP/PP layout, communication, general-purpose prefix caching (on for all requests).
- Same 2 GPUs only. Harness/dataset/gate never modified.

## Loopholes (forbidden)
- Caching/precomputing outputs for benchmark prompts; detecting the benchmark; tuning to seed 0 or these 400 prompts.
- Shrinking context/KV capacity the benchmark doesn't use.
- Changing temperature, thinking mode, output length, truncation.
- Non-exact speculation.
