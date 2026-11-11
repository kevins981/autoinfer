# Keep useful cache segments on both sides of a gap

**What it is.** Prefix-only caching abandons later reusable state when earlier blocks are evicted. AsymCache retains disjoint segments and recomputes the gaps. Its Multi-Segment Attention kernel processes these patterns together, while eviction weighs reuse probability against position-dependent recomputation cost.

**What it exploits.** Later tokens can cost more to reconstruct than early ones, even at equal hit rates. It helps repeated contexts under memory pressure; unique prompts or fully resident histories offer little gain.

**What it requires.** Cache-manager and scheduler changes plus a CUDA/CUTLASS attention kernel; approximately 6K Python and 2K C++/CUDA lines in vLLM.

**Exactness.** Reuse requires identical full preceding token history, not merely matching suffix text. Missing state is recomputed and all causal attention retained. Doubt: benchmarks force recorded output tokens, so they do not establish greedy-token equivalence independently.

**Reported results.** The source reports up to 1.86× TTFT reduction over vLLM-LRU for Llama-3.1-70B on four H20-96GB GPUs. LongBench/LooGLE experiments use 300 requests per dataset with varied arrival dispersion; 8B tests use one GPU.

**Sources.** Chunan Shi et al., [Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving](https://arxiv.org/html/2606.02964v1), June 1, 2026.
