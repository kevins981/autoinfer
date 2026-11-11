# Inference technique cards

Fifteen independent mechanisms; numbering is for reference, not ranking. Sources were opened and dates checked against November 10, 2024–November 10, 2026, inclusive. Dates below give first publication and, where applicable, the revision used. I conservatively excluded papers first public before the window even if conference publication came later. Engineering effort and applicability limits are qualitative assessments. Exactness refers to an unchanged model and precision with greedy decoding, allowing kernel/reduction rounding differences; unresolved implementation qualifications are stated on the relevant cards.

- [01-plan-attention-before-replay.md](01-plan-attention-before-replay.md) — Plan irregular attention before replaying fixed GPU work — Separate dynamic workload planning from a fixed graph executor. Source: January 2, 2025; revision April 21, 2025.
- [02-persistent-gpu-task-runtime.md](02-persistent-gpu-task-runtime.md) — Keep a task runtime running inside one GPU kernel — Dispatch dependent model tasks within a persistent device runtime. Source: December 22, 2025; revision June 10, 2026.
- [03-fuse-reduction-and-normalization.md](03-fuse-reduction-and-normalization.md) — Fuse communication with normalization between token groups — Normalize reduced token shards before gathering and overlap groups. Source: May 16, 2025; revision May 1, 2026.
- [04-group-offline-prefixes.md](04-group-offline-prefixes.md) — Schedule an offline job around shared prefixes — Use whole-job knowledge to preserve reuse and form productive token batches. Source: November 29, 2024; revision April 22, 2026.
- [05-map-work-to-a-memory-mesh.md](05-map-work-to-a-memory-mesh.md) — Move matrix tiles locally across a memory mesh — Map dense arithmetic and intact cache data to short on-chip paths. Source: February 6, 2025; revision May 30, 2025.
- [06-compute-attention-near-storage.md](06-compute-attention-near-storage.md) — Compute attention beside the stored cache — Move queries and compact results instead of transporting entire KV caches. Source: February 14, 2025; revision February 6, 2026.
- [07-separate-attention-from-experts.md](07-separate-attention-from-experts.md) — Give attention and experts separate device pools — Scale attention capacity independently from expert batching capacity. Source: April 3, 2025; revision July 26, 2025.
- [08-place-phases-with-network-flow.md](08-place-phases-with-network-flow.md) — Place inference phases using a network-flow model — Jointly plan heterogeneous device groups, parallel layouts, and cache transfers. Source: February 11, 2025.
- [09-prepare-the-next-batch-early.md](09-prepare-the-next-batch-early.md) — Prepare the next batch while the GPU runs — Hide host scheduling behind execution using deferred token dependencies. Source: December 4, 2024.
- [10-cache-activations-and-regenerate-kv.md](10-cache-activations-and-regenerate-kv.md) — Store activations and regenerate keys and values — Trade cache traffic for original projection work during otherwise idle cycles. Source: January 3, 2025; revision October 11, 2026.
- [11-verify-drafts-across-tokenizers.md](11-verify-drafts-across-tokenizers.md) — Verify draft text across different tokenizers — Align heterogeneous draft proposals and accept only target-verified greedy tokens. Source: January 31, 2025; revision June 11, 2025; ICML July 13–19, 2025.
- [12-tokenize-in-parallel-on-the-gpu.md](12-tokenize-in-parallel-on-the-gpu.md) — Tokenize text in parallel on the GPU — Parallelize encoding while preserving the exact reference token IDs. Source: October 9, 2026.
- [13-overlap-host-attention-with-device-work.md](13-overlap-host-attention-with-device-work.md) — Let host attention finish alongside later GPU work — Keep full-cache CPU attention productive through delayed synchronization. Source: June 3, 2025; revision January 14, 2026.
- [14-load-weight-files-before-making-tensors.md](14-load-weight-files-before-making-tensors.md) — Load weight-file groups before constructing tensors — Aggregate device transfers and construct tensor views after loading. Source: May 29, 2025.
- [15-send-queries-and-results-opposite-ways.md](15-send-queries-and-results-opposite-ways.md) — Send attention queries and results in opposite directions — Use both link directions to exchange queries and exact partial attention results. Source: December 29, 2024.

Techniques examined and left out:

- [POD-Attention](https://arxiv.org/abs/2410.18038) — First published October 23, 2024, before the window; later ASPLOS 2025 publication was not used to reset its date.
- [NEO](https://arxiv.org/abs/2411.01142) — First published November 2, 2024, before the window; its newer asynchronous extension is card 13.
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783) — First published November 4, 2024, before the window; card 15 supplies a different in-window communication mechanism.
- [SuffixDecoding](https://arxiv.org/abs/2411.04975) — First published November 7, 2024, before the window.
- [vAttention](https://arxiv.org/abs/2405.04437) — First published May 7, 2024, before the window.
- [TailorKV](https://arxiv.org/abs/2505.19586) — May 26, 2025; quantization and selective token loading are lossy despite “nearly lossless” task performance.
- [Compression-aware memory-controller design](https://arxiv.org/html/2503.18869v3) — March 24, 2025; its reversible bit-plane layout qualifies for exact storage, but reported latency experiments combine dynamic quantization and model/router changes. No isolated exact-form inference speedup was established, so it was left out of this small set.
