# Recompute part of a cache while transferring the rest

**What it is.** Offloaded decoding normally transfers complete KV tensors from CPU memory. KVPR instead transfers some saved layer-input activations and reconstructs their keys and values on the GPU. The remaining KV tensors transfer concurrently, with a profiler choosing the split.

**What it exploits.** Exchanges extra GPU arithmetic for less PCIe traffic when GPU compute would otherwise wait. It helps less when compute is saturated or the cache already fits locally.

**What it requires.** Saved activations, pinned host memory, profiling, and a transfer/recomputation scheduler; implementations build on existing offloading engines.

**Exactness.** Reconstruction repeats the original key/value projections from unchanged activations and weights. It neither approximates attention nor discards information, apart from permitted kernel rounding.

**Reported results.** The ACL paper reports 35.8% lower decoding latency than Hugging Face Accelerate for OPT-6.7B, batch 64, 128-token prompt/output, on A100-40GB with a 64-core AMD EPYC host. This is decoding latency, not whole-request acceleration.

**Sources.** Chaoyi Jiang, Lei Gao, Hossein Entezari Zarch, Murali Annavaram, [KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation](https://aclanthology.org/2025.findings-acl.997.pdf), July 2025; [submission history](https://arxiv.org/abs/2411.17089), November 26, 2024.
