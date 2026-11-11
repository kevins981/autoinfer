# Prepare the next batch while the GPU runs

**What it is.** A serial serving loop waits for GPU completion before scheduling and preparing another batch. SGLang v0.4 runs the CPU scheduler one batch ahead. Future-token placeholders and CUDA events defer resolution of token-dependent inputs while allocation and prefix-cache bookkeeping proceed.

**What it exploits.** CPU scheduling gaps between short GPU iterations. Small models and large tensor-parallel groups expose these gaps most; long compute-bound iterations already hide much host overhead.

**What it requires.** Dependency-aware asynchronous scheduling and synchronization inside the engine. Users receive it by default in this release; implementing it requires scheduler changes.

**Exactness.** A placeholder is resolved to the actual generated token before dependent GPU work uses it. It is not an accepted guess; all normal model operations still execute.

**Reported results.** The release reports 1.1× speedup over its previous scheduler. Its reproduction command uses Llama-3.2-3B-Instruct, 500 random requests, 4096 input and 2048 output tokens. GPU hardware is not specified alongside that scheduler benchmark; the separate router benchmark's eight A100s should not be assumed. Profiling shows no gaps with Triton, but a small gap with FlashInfer.

**Sources.** SGLang Team, [*SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs*](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), December 4, 2024, “Zero-Overhead Batch Scheduler.”
