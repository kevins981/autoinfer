# Let host attention finish alongside later GPU work

**What it is.** Offloading cache to CPU memory often creates synchronization bubbles. APEX keeps linear layers in one unified GPU batch, branches CPU/GPU attention, and delays collecting CPU results while independent GPU work continues. A profiled scheduler chooses this overlap or asymmetric sub-batch pipelining according to predicted execution times.

**What it exploits.** GPU cache-capacity pressure and CPU compute that can fit into a useful overlap window. When all requests fit on the GPU, APEX prefers GPU-only execution; slow CPUs and thread overhead limit benefit.

**What it requires.** CPU paged-attention kernels, pinned buffers, profiling, and a multithreaded runtime releasing Python's GIL. It extends NEO.

**Exactness.** CPU attention consumes the full offloaded cache. A request advances only when its required layer results are ready; delays do not substitute stale results. CPU/GPU kernels may round differently, without requiring reduced precision.

**Reported results.** The source reports 11% higher throughput than vLLM on Azure Conversation and 22% on LiveBench, for Llama3.1-8B on A10 with dual Xeon Gold 6342 CPUs. It reports up to 96% over vLLM on OSC with varied output lengths, Llama2-7B/T4 and dual Gold 6130. Latency includes queueing.

**Sources.** Jiakun Fan et al., [*APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs*](https://arxiv.org/html/2506.03296v4), first published June 3, 2025; revision January 14, 2026, §§3–5.
