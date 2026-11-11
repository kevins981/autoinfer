# Keep a task graph running on the GPU

**What it is.** Conventional inference launches separate kernels for successive operators. Mirage Persistent Kernel compiles the model into small GPU tasks connected by dependency events. A persistent kernel schedules ready tasks across GPU multiprocessors, allowing downstream work to start before an entire upstream operator finishes.

**What it exploits.** Removes launches, global operator barriers, and host scheduling gaps. Benefits shrink when large matrix operations already dominate execution.

**What it requires.** A compiler backend and device runtime; the prototype integrates with PyTorch and supports NVIDIA A100, H100, and B200.

**Exactness.** Tasks preserve the tensor program and dependencies. Same-precision fusion and reordered reductions can introduce the allowed rounding differences; no layers or tokens are approximated.

**Reported results.** The source reports Qwen3-8B decoding latency falling from 14.5 ms to 12.5 ms on A100 against optimized vLLM/SGLang. Its offline experiments use 64-token prompts, 1024-token outputs, BF16, and maximum batches 1–16. These are whole-runtime results.

**Sources.** Xinhao Cheng et al., [Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/html/2512.22219v1), December 22, 2025, §§3–6.
