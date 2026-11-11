# Decompress weights directly into matrix-multiply registers

**What it is.** Lossless compression usually expands weights into memory before multiplication. ZipServ packs common BF16 exponents into bitmaps and reconstructs complete weights directly in Tensor Core registers. It overlaps reconstruction with matrix multiplication during decoding, while prefill uses separate decompression and GEMM.

**What it exploits.** Saves weight bandwidth and intermediate memory traffic. Compute-heavy prefill benefits less, motivating the separate execution path.

**What it requires.** Offline packing, custom CUDA kernels, and loader/backend integration; the prototype contains approximately 3.5K lines. DFloat11 is a related entropy-coded alternative with block-level decompression.

**Exactness.** This is reversible encoding, not reduced numeric precision: every BF16 bit is restored. Multiplication still uses BF16 operands with ordinary kernel rounding.

**Reported results.** ZipServ reports 1.22× average throughput over vLLM across batches 8/32 and outputs 128–2048 tokens: Llama-3.1-8B on one RTX4090, Mistral-24B on two L40S, and Llama-3.1-70B on four L40S.

**Sources.** Ruibo Fan et al., [ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression](https://arxiv.org/html/2603.17435v1), March 18, 2026; Tianyi Zhang et al., [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float](https://arxiv.org/html/2504.11651v1), April 15, 2025.
