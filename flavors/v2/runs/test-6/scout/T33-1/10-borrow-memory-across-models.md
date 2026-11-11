# Let active models borrow idle models’ memory

**What it is.** Separate model servers often reserve GPU cache memory even while idle. Prism maps physical memory on demand beneath stable virtual addresses and coordinates allocation across models. Global placement and local request scheduling combine simultaneous sharing with model activation and eviction.

**What it exploits.** Bursty models need capacity at different times. Complementary demand helps; simultaneously busy models that exhaust memory leave little to borrow.

**What it requires.** CUDA virtual-memory APIs, GPU multiprocess sharing, and cross-model scheduling. The prototype adds an allocation library to SGLang and a substantial orchestration layer.

**Exactness.** Memory ownership and execution timing change, not model weights or token selection. Evicted state must be retained losslessly or recomputed before use; ordinary kernel rounding remains possible.

**Reported results.** The source reports serving eight models on two H100-80GB GPUs with up to 2.3× the request load of MuxServe++ at 99% TTFT SLO attainment on Hyperbolic traces. Broader tests span 58 models and up to 32 H100s; this measures load under latency constraints.

**Sources.** Shan Yu et al., [Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving](https://arxiv.org/html/2505.04021v1), May 6, 2025, §§5–7.
