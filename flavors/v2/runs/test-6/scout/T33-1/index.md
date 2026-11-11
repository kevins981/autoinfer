# Inference technique cards

Fifteen independent mechanisms; numbering is not a ranking. Dates identify the cited publication/version, with original submission dates where noted. Results are source-reported, not reproduced. Exactness is assessed against the same model, precision, inputs, and decoding constraints; explicit uncertainties remain in cards 14 and 15.

- [01-persistent-gpu-task-graph.md](01-persistent-gpu-task-graph.md) — Keep a task graph running on the GPU. Compile inference into dependency-driven tasks executed inside a persistent GPU kernel. Source date: 2025-12-22.
- [02-prepare-next-batch.md](02-prepare-next-batch.md) — Prepare the next batch while the GPU runs. Overlap next-step CPU scheduling with current-step GPU execution using deferred token dependencies. Source date: 2024-12-04.
- [03-rebalance-attention-tiles.md](03-rebalance-attention-tiles.md) — Rebalance attention tiles before graph replay. Replan attention work dynamically while retaining fixed buffers suitable for CUDA Graph replay. Source date: 2025-01-02.
- [04-weave-token-halves.md](04-weave-token-halves.md) — Overlap one token group with the other group’s communication. Split tokens into two efficient groups and weave computation, reduction, and normalization. Source date: 2025-05-16.
- [05-reshard-between-phases.md](05-reshard-between-phases.md) — Change model sharding between prefill and decoding. Use different parallel layouts for prefill and decode while amortizing state movement across offline work. Source date: 2025-03-09.
- [06-separate-attention-experts.md](06-separate-attention-experts.md) — Serve attention and experts from separate device pools. Aggregate routed expert work separately from attention and pipeline microbatches between device pools. Source date: 2025-04-03.
- [07-reorder-offline-requests.md](07-reorder-offline-requests.md) — Reorder offline requests around shared prefixes. Use the whole offline request set to schedule prefix reuse and fill mixed prefill/decode batches. Source date: 2024-11-29.
- [08-recompute-part-of-cache.md](08-recompute-part-of-cache.md) — Recompute part of a cache while transferring the rest. Rebuild selected keys and values from saved activations while concurrently loading the remaining cache. Source date: 2024-11-26; 2025-07.
- [09-checkpoint-recurrent-prefixes.md](09-checkpoint-recurrent-prefixes.md) — Checkpoint reusable recurrent states at selected boundaries. Cache complete recurrent snapshots selectively instead of attempting to roll overwritten state backward. Source date: 2024-11-28.
- [10-borrow-memory-across-models.md](10-borrow-memory-across-models.md) — Let active models borrow idle models’ memory. Redistribute physical GPU memory across model servers as their demand changes. Source date: 2025-05-06.
- [11-retrieve-token-drafts.md](11-retrieve-token-drafts.md) — Retrieve draft continuations from a suffix automaton. Use indexed text continuations as cheap proposals and accept only target-verified greedy tokens. Source date: 2024-11-16.
- [12-precompute-grammar-masks.md](12-precompute-grammar-masks.md) — Precompute most grammar checks and share parser stacks. Compile recurring grammar checks and share parser stacks while preserving an existing decoding constraint. Source date: 2024-11-22.
- [13-decompress-inside-matmul.md](13-decompress-inside-matmul.md) — Decompress weights directly into matrix-multiply registers. Restore losslessly compressed weights inside the consuming matrix kernel to save memory traffic. Source date: 2026-03-18; 2025-04-15.
- [14-parallel-tokenizer.md](14-parallel-tokenizer.md) — Tokenize independent text chunks across CPU threads. Parallelize tokenizer work and reuse CPU buffers; exact boundary handling needs validation. Source date: 2026-03-16.
- [15-reuse-disjoint-cache-segments.md](15-reuse-disjoint-cache-segments.md) — Keep useful cache segments on both sides of a gap. Recompute cache gaps while retaining expensive later segments under an unchanged causal history. Source date: 2026-06-01.

Techniques examined and left out:

- [vAttention](https://arxiv.org/abs/2405.04437) — outside the window: first submitted May 7, 2024, despite a 2025 conference publication.
- [NanoFlow](https://arxiv.org/abs/2408.12757) — outside the window: first submitted August 22, 2024. Card 02 describes a separately released December 2024 serving implementation, whose source acknowledges this antecedent.
- [NEO](https://arxiv.org/abs/2411.01142) — outside the window: first submitted November 2, 2024.
- [SuffixDecoding](https://arxiv.org/abs/2411.04975) — outside the window: first submitted November 7, 2024; also overlaps the retrieval-proposal mechanism in card 11.
- [HGCA](https://arxiv.org/html/2507.03153v1) — lossy: sparsifies CPU-side attention; “near-full attention quality” does not establish identical tokens.
- [HeadInfer](https://arxiv.org/html/2502.12574v1) — its exact form primarily demonstrates capacity expansion; a clean serving-time speed advantage was not established in the inspected results, and its faster sparse extension is lossy.
- [GPUTOK](https://arxiv.org/abs/2603.02597) — exactness unresolved: the abstract reports agreement with its CPU version but only similarity/overlap comparisons against reference tokenizers; the tokenizer mechanism also overlaps card 14.
- DFloat11 — consolidated into card 13 as a related reversible weight-compression approach rather than a second compression card.
- SGLang cache-aware routing — omitted to avoid another prefix-reuse card; cards 07, 09, and 15 cover different cache organization mechanisms. The dated primary release is linked in card 02.
