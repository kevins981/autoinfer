# Plan irregular attention before replaying fixed GPU work

**What it is.** Variable context lengths leave some attention workers busy after others finish. FlashInfer splits long contexts into chunks and assigns chunks to balance estimated work. A CPU planning pass writes queues and output mappings into fixed GPU buffers; the same plan serves every layer, while a captured CUDA graph executes it.

**What it exploits.** Uneven request lengths and repeated layer shapes. Uniform small contexts offer less balancing benefit; planning itself costs CPU time.

**What it requires.** An attention backend with separate `plan` and `run` interfaces, workspace buffers, and CUDA graph integration: library/runtime engineering.

**Exactness.** In the full-attention, unchanged-precision path, every required KV entry contributes. Partial attention results are combined with softmax normalization; only reduction order changes. Sparse pruning and FP8 variants are outside this card.

**Reported results.** The source's scheduler ablation reports inter-token latency of 8.63 ms with balancing versus 13.89 ms without it: Llama-3.1-8B-Instruct, H100 SXM5, SGLang, input lengths uniformly sampled from 4096–16384, output 256 tokens, request rate 1.

**Sources.** Zihao Ye et al., [*FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*](https://arxiv.org/html/2501.01005v2), first published January 2, 2025; revision April 21, 2025, §§3.3–3.4 and Appendix G.3.
