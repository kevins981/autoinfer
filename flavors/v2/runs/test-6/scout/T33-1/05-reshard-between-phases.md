# Change model sharding between prefill and decoding

**What it is.** A fixed distributed layout must serve both prompt processing and token generation. Seesaw changes the model's tensor/pipeline parallel layout between these phases. It groups work to reduce transitions and buffers KV state through a CPU tier while rearranging execution.

**What it exploits.** Prefill and decoding favor different parallel layouts. Large offline workloads amortize transitions; short jobs or latency-sensitive streaming may not.

**What it requires.** A distributed inference runtime with model re-sharding, host cache buffers, asynchronous transfers, and transition-aware scheduling.

**Exactness.** Re-sharding moves unchanged weights and cache values between devices. Computation is repartitioned rather than pruned; only same-precision reduction-order differences are expected.

**Reported results.** The source reports up to 1.78× throughput, averaging 1.36×, against vLLM 0.5.4. Tests use A10, L4, and A100 GPUs; 15B Llama-3-derived, CodeLlama-34B, and Llama-2-70B models in FP16; and offline sets of 2000 ShareGPT or 500 arXiv-summarization requests.

**Sources.** Qidong Su et al., [Seesaw: High-throughput LLM Inference via Model Re-sharding](https://arxiv.org/html/2503.06433v1), March 9, 2025, §§4–6.
