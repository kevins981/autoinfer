# Fuse communication with normalization between token groups

**What it is.** Tensor-parallel layers normally finish an all-reduce, then redundantly normalize the result on every GPU. TokenWeave instead normalizes each reduced token shard before gathering it, fusing communication and normalization. It splits tokens into groups so one group's communication overlaps another's compute, using hardware multicast/reduction support.

**What it exploits.** Exposed collective latency and competition between communication and compute for GPU resources. Tiny groups lose to splitting overhead; a single-device model has no collective to hide.

**What it requires.** Tensor-parallel runtime changes and custom kernels; the demonstrated path uses Hopper NVSHARP/Multimem and symmetric memory.

**Exactness.** All tensor-parallel contributions still enter the reduction, and the normalization remains mathematically equivalent. Collective and fused-kernel rounding may differ. No stale activations replace current ones.

**Reported results.** The source reports up to 1.28× latency speedup and 1.19× throughput on eight H100s against vLLM V1 0.8.5. Models include Llama-3.3-70B, Qwen2.5-72B, and Mixtral-8x22B, with ShareGPT/arXiv traces and fixed-length workloads; efficient overlap extends down to 1024 tokens per iteration.

**Sources.** Raja Gond, Nipun Kwatra, and Ramachandran Ramjee, [*TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference*](https://arxiv.org/html/2505.11329v5), first published May 16, 2025; revision May 1, 2026, §§4–5.
