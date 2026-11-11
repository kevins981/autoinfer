# Keep a task runtime running inside one GPU kernel

**What it is.** Conventional inference launches separate kernels for operators, even with graph replay. MPK compiles operators into small tasks connected by data-dependency events. GPU schedulers dispatch ready tasks to persistent workers, allowing adjacent operators and communication to overlap without returning control to the host.

**What it exploits.** Kernel boundaries, dispatch overhead, and pipeline bubbles, especially at small decode batches. Large compute-saturating batches leave less overhead to recover.

**What it requires.** A compiler, device queues, dependency tracking, and specialized task implementations. Adoption uses MPK; implementing the mechanism requires a new runtime.

**Exactness.** Tasks preserve the model's dependencies and arithmetic at unchanged precision. Fusion and reduction order can introduce the permitted rounding differences; no layers or selected experts are omitted.

**Reported results.** The source reports Qwen3-8B per-token decode latency of 12.5 ms versus 14.5 ms for optimized vLLM/SGLang on A100. Its BF16 offline evaluation uses batch sizes 1–16, 64-token prompts and 1024-token outputs across A100, H100, and B200; gains are strongest at batch one.

**Sources.** Xinhao Cheng et al., [*MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*](https://arxiv.org/html/2512.22219v2), first published December 22, 2025; revision June 10, 2026, §§3–6.
