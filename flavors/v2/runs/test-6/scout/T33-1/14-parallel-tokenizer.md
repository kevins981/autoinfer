# Tokenize independent text chunks across CPU threads

**What it is.** A serial tokenizer can delay a GPU even when most prompt state is cached. Fastokens parallelizes regex scanning using overlapping authority zones, then BPE-encodes independent chunks in a thread pool. Thread-local/shared caches and reusable buffers reduce synchronization and allocation.

**What it exploits.** Long, frequently cached prompts expose CPU tokenization as a large part of first-token latency. Short prompts or GPU-dominated prefill offer smaller gains.

**What it requires.** A Rust tokenizer replacement, model-compatible tokenizer data, and multicore CPUs; integrated with Dynamo/SGLang.

**Exactness.** Intended as a drop-in implementation of existing tokenization rules, without retraining. Doubt: the blog provides no general token-equivalence proof; its fixed regex overlap requires boundary validation for each supported tokenizer. Identical token IDs are necessary, not merely similar generated text.

**Reported results.** Crusoe/NVIDIA report 9.1× average tokenization speedup over HuggingFace across four model tokenizers, ShareGPT/LongBench-v2, lengths 512–100K, and Xeon 8468V/8568Y and Grace CPUs. Reported TTFT improvement reaches 40% in Dynamo/SGLang tests on HGX B200/GB200 NVL72, batch 1, using DeepSeek V3 and GPT-OSS-120B.

**Sources.** Alon Kejzman et al., [Reducing TTFT by CPUMaxxing Tokenization](https://www.crusoe.ai/resources/blog/reducing-ttft-by-cpumaxxing-tokenization), March 16, 2026.
