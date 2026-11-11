# Schedule an offline job around shared prefixes

**What it is.** Online cache policies can evict a prefix before another request needs it. BatchLLM inspects the whole pending job, explicitly groups identical prefixes, and schedules each group together. It reorders groups and mixes decode tokens with later prefill chunks to keep large token batches productive.

**What it exploits.** Advance knowledge of many requests with repeated instructions or documents. Unique prompts remove reuse; online arrival and tight individual deadlines restrict reordering.

**What it requires.** Global prefix indexing, a batch scheduler, and a shared-prefix attention backend, integrated into vLLM with fallback kernels.

**Exactness.** Reuse applies only to identical causal token prefixes under the same model and settings. Scheduling independent requests changes no dependencies within a request; shared attention can change reduction rounding. The mechanism requires no quantization.

**Reported results.** The source reports 1.7× speedup over the best vLLM configuration and 1.5× over SGLang for Llama-3.1-8B on A100: sharing degree 2, common prefix 2000 tokens, distinct context 200, output 100, 6400 requests. These are end-to-end throughput measurements. Separate 70B tests use GPTQ and are excluded here.

**Sources.** Zhen Zheng et al., [*BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching*](https://arxiv.org/html/2412.03594v3), first published November 29, 2024; revision April 22, 2026, §§3–4.
