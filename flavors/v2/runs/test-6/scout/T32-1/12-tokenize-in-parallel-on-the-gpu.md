# Tokenize text in parallel on the GPU

**What it is.** CPU tokenization can remain on the critical path after prefix caching removes most prefill. GraphTok evaluates pre-token boundaries in parallel and cooperatively finds BPE merges while preserving their reference order. A replayable CUDA graph runs the encoding pipeline; a dispatcher sends short inputs to a CPU tokenizer.

**What it exploits.** Long fresh text, concurrent inputs, and CPU frontend saturation. Short inputs can lose to GPU launch/transfer overhead; cold-cache model prefill can dwarf tokenization.

**What it requires.** Family-specific tokenizer frontends, CUDA kernels/graphs, and a vLLM adapter or Dynamo backend: substantial library engineering.

**Exactness.** The contract is identical ordered token IDs, including normalization and special-token behavior, for supported configurations. Unsupported structures are rejected. The paper checks against reference encoders; this claim is narrower than supporting arbitrary tokenizer files.

**Reported results.** On H100 NVL with dual Xeon Gold 6336Y, the source reports 1.7–3.2 ms for one million GPT-2 tokens including transfers, and 4.7–6.0× over gigatoken on Llama3 tokenization. These are tokenizer measurements, with no LLM/batch-serving speedup implied. Its fleet-capacity experiment simulates backend GPUs.

**Sources.** Mohammad Siavashi et al., [*Taking Tokenization off the Host for Agentic LLM Serving*](https://arxiv.org/html/2610.13325v1), October 9, 2026, §§4–6.
