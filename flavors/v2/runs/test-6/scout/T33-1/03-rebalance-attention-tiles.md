# Rebalance attention tiles before graph replay

**What it is.** Assigning one attention request to each worker leaves workers idle when sequence lengths differ. FlashInfer splits long attention jobs into chunks and assigns tiles using estimated cost. A planning phase writes assignments into fixed-address buffers, letting the execution phase replay a CUDA Graph despite changing request lengths.

**What it exploits.** Removes uneven attention workloads and launch overhead together. Uniform, already well-balanced batches offer less benefit.

**What it requires.** CUDA attention kernels, workspace buffers, and separate plan/run APIs integrated into the serving loop.

**Exactness.** Use full attention at the original precision. All chunks contribute through softmax-aware merging; deterministic aggregation avoids atomic-order nondeterminism. Sparse and reduced-precision options are excluded here.

**Reported results.** The paper reports 29–69% lower inter-token latency for its complete backend against SGLang's Triton backend: Llama-3.1-8B on one H100 and 70B on four H100s, ShareGPT and variable-length online requests. This does not isolate scheduling's contribution.

**Sources.** Zihao Ye et al., [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/html/2501.01005v1), January 2, 2025, §§3.3–4.1.
