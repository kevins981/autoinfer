# Reorder offline requests around shared prefixes

**What it is.** Streaming schedulers see requests incrementally and may evict prefixes just before reuse. BatchLLM examines an entire offline job and groups requests with identical prefixes. It orders groups to mix decoding with subsequent prefill chunks, then admits work according to available KV memory.

**What it exploits.** Foreknowledge of the request set prevents avoidable cache misses and poorly filled batches. It offers less when prefixes differ or requests must finish in arrival order.

**What it requires.** Prefix-group preprocessing, a modified vLLM scheduler, and a prefix-sharing attention backend; the prototype uses Python and Triton.

**Exactness.** Reuse requires identical token prefixes with identical model context. Scheduling changes completion order, not per-request causal computation; no approximate cache blending is involved.

**Reported results.** Version 1 reports 1.1–1.5× speedup on A100 and up to 2× on MI200 against vLLM with prefix caching and chunked prefill. Microbenchmarks include Llama3-8B, Qwen2-7B, and Mistral-7B with varying shared-prefix proportions. Results cover the complete system.

**Sources.** Zhen Zheng et al., [BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching](https://arxiv.org/html/2412.03594v1), first submitted November 29, 2024, §§4–6.
