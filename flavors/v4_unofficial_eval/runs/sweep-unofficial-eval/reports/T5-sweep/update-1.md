# T5 map, update 1 (closing; run wrapping up). HEAD = T3-004

## Since 6-ranking
| item | record / job | result |
|---|---|---|
| C1 lossless AR encoding: ruling | manager, 2026-11-15 (task_definition.md) | ALLOWED with a bitwise stress test (-0, subnormals, inf/NaN, raw fallback) and a generic codec |
| C1 step A: compressibility of real AR payloads | T5-001, jobs 206 (+ re-dump) | B=32 / 4-bit offset: 0.77 of raw, 0.2-0.6% raw blocks, round-trip bitwise on 160 files; 3-bit: 0.83 (41% raw). Gate (<= 0.82) passed provisionally on thin data |
| C1 step B: compressed host-AR kernel + microbench | - | not started (run ended) |
| interface with T4's P4 | messages with leader-4 | C1 = drop-in HostAllReduce variant (same all_reduce(inp)->out, graph-capturable, constant grid, per-instance buffers). Under P4 the link is the bottleneck, so the byte saving should carry over almost 1:1 |

## Verdicts changed
- None after 5-compare. (The corrections are in 5-compare §2: mixed-step CPU overstated; no PCIe duplex; in_proj_ba overlapped.)

## Current ranking (for the next team)
1. **C1 step B**: build the encode-on-write / decode-on-read host AR (bitwise vs current), microbench at 48 / 300 / 1024 rows
   vs current and NCCL, then extend the host AR to prefill sizes (replacing NCCL) and combine it with T4's P4. Napkin ~4%
   alone; more under P4. Riskiest: whether encode/decode inside the AR kernel stays under ~1-2 us at M=48, and whether
   link time keeps scaling with bytes at 12-bit granularity.
2. Decode W8A16 kernel redesign (~3-4%; T4's #3, nobody took it).
3. lm_head Triton GEMV (~0.8%; T4's #4).
4. Full CUDA graph for mixed steps: ≤ 1-2% (T2-014 shows the CPU-bound part is mostly overlapped).
