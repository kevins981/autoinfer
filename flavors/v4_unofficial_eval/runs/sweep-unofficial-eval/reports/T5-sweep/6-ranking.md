# T5 sweep, step 6: ranking (vs HEAD T3-004)

Time base (corrected per 5-compare §2): decode steps ~62% of wall (pure-16 step 27.6 ms, of which host
all-reduce 5.7 ms = 135 x 42 us); mixed-step extra ~38% (T3 job 198 / T4 job 199), of which
communication ~0.1 ms per prompt token (~10% of wall), NCCL at ~80 us/MB.

| rank | candidate | napkin gain | wins if | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | **C1: lossless block-exponent encoding of the all-reduce payload** (sign + 4-bit exponent offset from a per-32-value block max + 7-bit mantissa; raw fallback blocks; bit-exact), in the host AR kernel; then the same compressed host AR for prefill sizes instead of NCCL | decode: ~23% of the AR's bandwidth part (~34 of 42 us) ≈ -6..-8 us x 135 = **-0.8..-1.0 ms/step (~2%)**; mixed: replace NCCL (~80 us/MB) by compressed host AR (~60 us/MB): **~-2.4%**; total **~4%** (bitwise-identical outputs) | the ruling allows lossless encoding (it is not a precision change); real payloads compress to ≤ 0.8 (raw-block rate < ~5%); pack/unpack adds ≤ 1-2 us at M=48; link time stays ∝ bytes at 12-bit granularity | the exponent spread within 32 neighbouring channels of real o_proj/down/out_proj partial sums (outlier channels, near-zero values) | (a) manager ruling (free); (b) dump ~100 real AR inputs from one HEAD server run and compute the exact compressed size offline (~2 GPU-min); (c) microbench compressed vs current host AR at M=48 / 300 rows, 2 GPUs, CUDA-graph timed (~2 GPU-min) |
| 2 | decode W8A16 new kernel design (single M-block of 48, split-K, deeper cp.async) | -1..-1.5 ms/step, 3-4% | isolated BW > 0.9 TB/s carries in situ | T1's 0.77-0.90 ceiling is the card, not the kernel family | isolated microbench of gate_up at M=48 (T4's fallback #3: coordinate, don't duplicate) |
| 3 | GEMM-free CPU trims in mixed steps (full CUDA graph of the mixed step, or fused GDN prefill glue) | ≤ ~10 ms CPU per mixed step, mostly overlapped: **≤ 1-2%** | the CPU-bound pieces pace the step (T2-014 says mostly not) | that any of it is on the critical path | no-nsys A/B of T2-014-like trims was already null: needs a new argument first |
| 4 | lm_head bf16 Triton GEMV (0.80 -> 0.92 TB/s) | -0.2..-0.3 ms/step, ~0.8% | in-situ = isolated | T1-014's in-situ loss | microbench + quick bench (T4's #4: coordinate) |
| 5 | smaller draft head vocab (48-64K ids by BPE order, generic text) | 0.17-0.25 ms/step saved vs acceptance loss | coverage on generic text stays > 99% | acceptance loss on chat text | generic-text coverage count (CPU-only) |
| 6 | mixed-step async (no blocking `seq_lens.cpu()`) | ≤ 0.5% | | | - |

Dropped (5-compare §1): pipelined/duplex AR, in_proj_ba, 3 drafts, tree, lazy K=3, PP/DP, decode dual-batch, admission batching on benchmark data.

**Plan (72.7 GPU-min):** ask the manager for the C1 ruling now; meanwhile a worker builds the offline
compressibility check (dump real AR payloads from one HEAD run, ~2 min) and the kernel. Only a ratio ≤ 0.8 gets
the kernel microbench (~2 min), then an engine record and unofficial_eval (~5 min each). Coordinate #2/#4 with T4.
