# Assumption ledger (run review-eval) — DRAFT

## Frozen
- Model Qwen/Qwen3.8-27B-FP8, weights, architecture, tokenizer, chat template.
- FP8 weights, FP8 KV cache; activations bf16 or the checkpoint's own dynamic group-128 FP8 scheme.
- Harness (2x400 ShareGPT, c=16, temp 0, seed 0), 128K long-request gate, GSM8K-250 >= 89.6%,
  latency guard (+20% on TTFT/TPOT p99 vs BASE).
- Prefix caching on. General OpenAI server (tool/reasoning parsers). Max context >= 131072.
- Hardware: 2x RTX 4090, no P2P, driver 580 / CUDA 13.0, vLLM 4527515cd7 as the starting code.

## Open
- Engine code, kernels, build; TP/PP/other splits; serve flags; CUDA graphs; KV budget; scheduler.
- Speculative decoding method and draft count (exact verification).
- Pending human: drafter approximations [Q1], extra draft models [Q2], promotion rule [Q3].
