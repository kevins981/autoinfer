# Load weight-file groups before constructing tensors

**What it is.** Conventional loaders deserialize individual tensors in host memory and then copy them to GPUs. Fastsafetensors first transfers groups of file data into device buffers and constructs tensor views there. Parallel reads, GPU sharding, and peer-to-peer transfers avoid repeated small copies and host preprocessing.

**What it exploits.** Cold starts, model switching, and tensor-by-tensor loading overhead. A warm resident model gains no decode speed; slow storage or an unfavorable NUMA topology can dominate.

**What it requires.** A safetensors loader integration and appropriate storage/device paths; GPUDirect Storage is optional. The paper's vLLM integration changed 42 Python lines.

**Exactness.** Weight bytes and shapes are preserved. Deserialization and partition placement change, with no quantization or model conversion required. Tensor lifetime and distributed collective order still need correct handling.

**Reported results.** Table II reports vLLM startup of 3.67 s versus 7.29 s for Llama2-7B on one A100, and 4.74 s versus 12.39 s for Llama2-13B on four L40S GPUs. These are startup measurements, not tokens/s; request batch/load is inapplicable. The evaluated vLLM snapshot is March 3, 2025.

**Sources.** Takeshi Yoshimura et al., [*Speeding up Model Loading with fastsafetensors*](https://arxiv.org/html/2505.23072v1), May 29, 2025, §§III–IV, especially Table II.
