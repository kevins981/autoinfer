# Speed of light: Qwen3.8-27B-FP8 on 2x RTX 4090 (TP=2), official benchmark

Physical lower bounds on time for this task, from the model config, the hardware specs and the
workload. Not measurements, and nothing about any implementation. Use them to see where the
largest gaps between a design and the hardware could be. Every number is approximate; the inputs
marked (check) are worth verifying before you rely on them.

## Inputs
| | Value | Source |
|---|---|---|
| Layers | 64: 48 linear-attention (GDN) + 16 full attention (every 4th); hidden 5120; MLP 17408 | config.json |
| Weights read per GPU per forward | ~12.2 GB (FP8, TP=2 halves) | config shapes |
| Target output layer (lm_head) | 248,320 x 5120, bf16: ~1.27 GB per GPU | config (vocab, untied) |
| Draft head (MTP) | 1 layer (~0.2 GB per GPU, FP8) + an output layer; 2 draft steps per decode step | config, serve args |
| GDN recurrent state | 48 heads x 128 x 128 per layer per request: ~1.6 MB per GPU if fp32 (check dtype); x48 layers x16 requests = ~1.2 GB per GPU | config shapes |
| DRAM bandwidth | ~1.0 TB/s per GPU (spec 1008 GB/s) | spec |
| FP8 tensor throughput | ~165 TFLOP/s with fp32 accumulate, ~330 with fp16 accumulate (dense, spec; see the accumulation rule in TASK.md) | spec |
| GPU-GPU link | none direct; via host memory over PCIe 4.0 x16: ~25-31 GB/s per direction | spec (check) |
| Workload | 400 requests, ~260 prompt tokens and ~228 output tokens each; 16 concurrent | TASK.md |
| Speculation | output tokens per sequence per decode step = accepted drafts + 1 (check the champion's official metrics) | |

## Decode step (16 sequences, 3 rows each with 2 drafts: M = 48)
| Cost | Bytes or work per GPU | Floor |
|---|---|---|
| Body weights | 12.2 GB | ~12.2 ms |
| Target lm_head | 1.27 GB | ~1.3 ms |
| Draft head, 2 steps | ~0.4 GB (small output layer) to ~3 GB (full bf16 lm_head each step) | ~0.4-3 ms |
| GDN state | read ~1.2 GB, write back up to ~1.2 GB (if fp32) | ~1.2-2.4 ms |
| KV cache (16 attention layers, fp8) | ~0.1-0.3 GB | ~0.2 ms |
| Compute | 48 rows x ~25.6 GFLOP = ~1.2 TFLOP | ~4-7 ms: below the memory time, so memory-bound (ridge at ~80-160 rows) |
| Communication | 2 all-reduces per layer = 128; each 48 x 5120 bf16 = ~0.5 MB per direction | ~2.4 ms if both directions overlap fully, ~4.8 ms if not; zero on the critical path only if hidden behind other work |

**Floor: ~15 ms per decode step** with everything else ideal and communication hidden, ~17-20 ms if
it isn't.

## Prompt processing (prefill)
~104K prompt tokens x ~25.6 GFLOP per GPU = ~2.7 PFLOP per GPU: **~16 s at 165 TFLOP/s** (~8 s at
330). Its all-reduces move ~10 KB per token per all-reduce: ~5 s of link time per GPU if not
hidden. Prefill is compute-bound.

## Whole benchmark
~91K output tokens at ~37 per decode step (16 x 2.33) = ~2,450 decode steps.
| | Decode | Prefill | Total | tok/s |
|---|---|---|---|---|
| Optimistic floor | 2,450 x 15 ms = ~37 s | ~8-16 s | ~45-53 s | ~1,700-2,000 |
| Communication not hidden | 2,450 x ~19 ms = ~47 s | ~16-21 s | ~63-68 s | ~1,350-1,450 |

Compare with the champion's official number (champion.md).

## The levers these numbers point to
- **Tokens per decode step** multiply everything: the step's cost is mostly fixed (weights), so
  more accepted tokens per step (better drafts, more of them verified cheaply) scale throughput
  almost directly. No hardware floor bounds acceptance.
- **Bytes per step:** weights, lm_heads, recurrent state. Exact representations only (TASK.md).
- **Communication on the critical path:** 128 dependent exchanges per step over a slow link.
- **Prefill compute:** near-compute-bound; the accumulation rule sets its tensor throughput.
