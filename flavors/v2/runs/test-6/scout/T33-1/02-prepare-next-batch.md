# Prepare the next batch while the GPU runs

**What it is.** A serial serving loop waits for GPU execution before preparing the next batch. SGLang v0.4 instead runs CPU scheduling one batch ahead. Future-token placeholders and CUDA events resolve dependencies when the real tokens become available.

**What it exploits.** Hides batch construction, prefix lookup, and memory-management overhead behind GPU computation. Gains diminish when those CPU costs are already hidden or insignificant.

**What it requires.** An asynchronous scheduler with careful event synchronization; enabled by default in that release, but a runtime change to reproduce elsewhere.

**Exactness.** Placeholders carry dependencies, not guessed output tokens. Actual GPU results supply the next step's inputs; weights, attention, and greedy selection remain unchanged.

**Reported results.** The SGLang team reports 1.1× speedup over its previous scheduler, especially for small models and larger tensor-parallel groups. Its reproduction uses Llama-3.2-3B-Instruct, 500 requests, 4096-token inputs, and 2048-token outputs, comparing overlap enabled/disabled. GPU hardware is not specified in the article's benchmark text.

**Sources.** The SGLang Team, [SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), December 4, 2024.
