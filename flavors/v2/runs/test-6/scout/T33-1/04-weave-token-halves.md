# Overlap one token group with the other group’s communication

**What it is.** Tensor-parallel inference commonly alternates computation with blocking reductions. TokenWeave splits a batch's tokens into two groups sized to avoid wasting GPU execution waves. It overlaps one group's computation with the other's communication, and fuses reduction with RMS normalization.

**What it exploits.** Uses otherwise idle links and hides memory-bound normalization behind computation. Tiny token groups or negligible communication leave little to overlap.

**What it requires.** Runtime scheduling changes and a fused collective kernel. The demonstrated implementation uses Hopper Multimem/NV-SHARP support on NVLink-connected H100s.

**Exactness.** Independent tokens are rescheduled; collective sums and normalization still operate on the same values. The algebraic reordering preserves computation at the original precision, subject to reduction rounding.

**Reported results.** The source reports up to 29% latency improvement and 26% throughput improvement across workloads. Its Llama-3.3-70B example uses an eight-H100 DGX and compares against vLLM with an optimized Multimem reduction across sequence lengths, including 18% latency improvement at 1K tokens.

**Sources.** Raja Gond, Nipun Kwatra, Ramachandran Ramjee, [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/html/2505.11329v1), May 16, 2025.
