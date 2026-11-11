# Move matrix tiles locally across a memory mesh

**What it is.** Shared-memory-style algorithms make distant accesses expensive on a wafer of small cores. WaferLLM interleaves matrix-tile placement so cyclic exchanges use short physical paths, and uses tree reductions for matrix-vector products. It remaps weights between prefill and decode and shifts KV entries between neighboring cores to balance storage.

**What it exploits.** Distributed on-chip SRAM and fast neighbor links, with little memory per core. Ordinary GPUs do not expose this memory mesh; oversized models still need off-chip capacity.

**What it requires.** Wafer-scale hardware and a custom mapping, kernel, and cache-management stack; demonstrated on Cerebras WSE-2.

**Exactness.** The transformations rearrange dense arithmetic and relocate intact cache entries. No approximation is intrinsic. The paper does not establish token-for-token parity across hardware, so unchanged numeric precision and kernel semantics remain an implementation qualification.

**Reported results.** The source reports 10–20× full-inference speedup over an A100 cluster running SGLang, for Llama3-8B/Llama2-13B and multiple input/output lengths, measuring per-request performance. This combines different hardware and the entire stack. Llama2 context-limit modifications in some experiments are outside this card.

**Sources.** Congjie He et al., [*WaferLLM: Large Language Model Inference at Wafer Scale*](https://arxiv.org/html/2502.04563v3), first published February 6, 2025; revision May 30, 2025, §§4–7.
