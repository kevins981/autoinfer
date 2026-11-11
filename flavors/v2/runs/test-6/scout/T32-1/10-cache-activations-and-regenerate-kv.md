# Store activations and regenerate keys and values

**What it is.** Offloaded attention repeatedly transfers two cached tensors, keys and values, across PCIe. CAPTURE stores some blocks as the layer's input activations instead, regenerating their keys and values with the original projections. It mixes these blocks with ordinary KV blocks and overlaps regeneration with weight loading.

**What it exploits.** Spare GPU arithmetic during transfer-bound inference. Activation storage saves space for conventional multi-head attention; grouped-query layouts can weaken that saving, and compute-bound execution may lose to recomputation.

**What it requires.** A hybrid block allocator, modified attention kernels, and an asynchronous offloading engine. The source reports about 12K lines of changes to vLLM.

**Exactness.** Stored activations and original weights reconstruct the same projections at unchanged precision, apart from kernel rounding. This is recomputation, not lossy cache compression; no model layer is omitted.

**Reported results.** The source reports 2.19× throughput over FlexGen. Evaluation uses FP16 OPT-6.7B/13B/30B/66B, RTX 4090 24GB, dual Xeon Gold 6326 and 882GB host memory; batch 128, 128 output tokens, varied prompts. Throughput includes prefill and decode.

**Sources.** Sanghyeon Lee et al., [*Throughput-Oriented LLM Inference via KV-Activation Hybrid Caching with a Single GPU*](https://arxiv.org/html/2501.01792v2), first published January 3, 2025; revision October 11, 2026, §§III–V.
