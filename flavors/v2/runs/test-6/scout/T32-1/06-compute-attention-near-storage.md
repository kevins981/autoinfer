# Compute attention beside the stored cache

**What it is.** SSD offloading normally streams cached keys and values back to a CPU or GPU for attention. HILOS places attention accelerators beside storage, sending queries toward the cache and returning compact attention results. Host caching and delayed writeback further reduce transfers and hide writes.

**What it exploits.** Long-context offline batches whose KV traffic saturates host links or exceeds DRAM. It offers less benefit when the cache fits near the GPU or storage-side compute is slower than transferring it.

**What it requires.** SmartSSDs with FPGA/DRAM, attention hardware, and coordinating middleware: a hardware/runtime co-design.

**Exactness.** The design retains the complete cache, with FP16 storage and FP32 intermediate arithmetic. Moving attention is exact in principle. The implementation's finite padding-mask sentinel and FPGA exponential behavior need equivalence checking; task accuracy alone does not prove identical greedy tokens.

**Reported results.** The source reports 5.3–7.8× decoding throughput over storage-offloaded FlexGen for long contexts where DRAM offloading fails. Tests use up to 16 SmartSSDs plus A100/H100, OPT models, default batch 16 and 64 output tokens. Some context lengths exceed model training limits; those are performance experiments.

**Sources.** Hongsun Jang et al., [*A Cost-Effective Near-Storage Processing Solution for Offline Inference of Long-Context LLMs*](https://arxiv.org/html/2502.09921v2), February 6, 2026; initial INF² preprint February 14, 2025, §§4–6.
