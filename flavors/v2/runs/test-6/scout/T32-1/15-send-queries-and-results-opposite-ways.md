# Send attention queries and results in opposite directions

**What it is.** Ring attention typically circulates keys and values in one direction. TokenRing keeps local KV blocks and concurrently sends queries one way and partial outputs with softmax normalization statistics the other way. It combines block results and uses zigzag sequence placement to balance causal attention work.

**What it exploits.** Bidirectional link capacity otherwise left idle, especially with many sequence-parallel devices. Few devices, limited connectivity, or communication contention reduce the advantage.

**What it requires.** Sequence-parallel attention, asynchronous peer-to-peer transfers, and topology-aware orchestration. The prototype uses PyTorch/NCCL and FlashAttention; full-mesh connectivity is the intended fit.

**Exactness.** All causally required attention blocks participate, and partial outputs are merged using their normalization statistics. This reorganizes full attention, with reduction-order rounding differences; no tokens or cache entries are approximated.

**Reported results.** On four PCIe-connected NVIDIA A10s, Llama2-7B attention dimensions, sequence length 24,000, the source reports approximately 3.5 ms for steps 0/1 and 4.6 ms for step 2, versus about 7.6 ms per Ring Attention round. These are attention-step profiles, not complete-model speedups; request batch is unspecified.

**Sources.** Zongwu Wang, Fangxin Liu, Mingshuai Li, and Li Jiang, [*TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication*](https://arxiv.org/html/2412.20501v1), December 29, 2024, §§3–4.
