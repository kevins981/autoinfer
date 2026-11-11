# Checkpoint reusable recurrent states at selected boundaries

**What it is.** Transformer prefix caches can reuse earlier positions by slicing token-indexed state. Hybrid attention/recurrent models overwrite recurrent state, so earlier boundaries need separate snapshots. Marconi admits selected snapshots and evicts according to recency and compute saved per byte, rather than checkpointing every small block.

**What it exploits.** Avoids filling memory with large, seldom-reused recurrent snapshots. It needs repeated exact prefixes; unrelated requests or abundant cache memory reduce the benefit.

**What it requires.** A hybrid-state radix cache with new admission/eviction policies in the serving runtime. It serves an existing hybrid model without modifying it.

**Exactness.** Hits restore the complete recurrent state and attention cache at the exact matching boundary. Missing boundaries are recomputed; a later recurrent state is never treated as an earlier one.

**Reported results.** The source reports up to 71.1% lower P95 time-to-first-token, or 617.0 ms, against baseline prefix caching. Latency evaluation serves Jamba-1.5-Mini on four A100-40GB GPUs in FP16; workload studies include LMSys, ShareGPT, and SWE-Bench.

**Sources.** Rui Pan et al., [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/html/2411.19379v1), November 28, 2024, §§4–5.
