# Place inference phases using a network-flow model

**What it is.** A heterogeneous fleet cannot use one uniform parallel layout efficiently. HexGen-2 represents GPUs and their links as a weighted graph, partitions them into prefill/decode groups, and uses maximum flow to plan work and KV transfers. Iterative swaps refine the grouping alongside tensor/pipeline parallel layouts.

**What it exploits.** Different GPU compute, memory bandwidth, capacity, and link speeds. Uniform hardware has fewer placement opportunities; inaccurate profiles or changing traffic can make a static plan less useful.

**What it requires.** Hardware/link profiling, a placement optimizer, a coordinator, and asynchronous cache-transfer support: a distributed runtime.

**Exactness.** Placement optimization approximates the scheduling problem, not model computation. Identical weights and full KV state move between phases; only partitioned arithmetic rounding can differ. No precision reduction is required.

**Reported results.** The source reports up to 2× and average 1.3× throughput versus DistServe at comparable price budgets; versus HexGen, up to 1.5× and average 1.4×. OPT-30B and Llama2-70B run on RunPod H100/A100/L40/A6000 configurations, using Azure-derived workloads with varied prefill/decode lengths, online and saturated offline loads.

**Sources.** Youhe Jiang, Ran Yan, and Binhang Yuan, [*HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment*](https://arxiv.org/html/2502.07903v1), February 11, 2025, §§3–5.
