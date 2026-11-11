# T1-042: the full stack (self-contained description)

Official: 942.2 tok/s at c=16 (trials 941.4 / 942.9), +48.2% vs BASE 631.4 (2-trial BASE 635.9). GSM8K 93.2%, 128K request pass, latency gate pass. bank/T1-042/official/1/.

## How to reproduce
- Apply `change.diff` to the task baseline (`$CACHE_ROOT/tasks/<task>/baseline/vllm`). It is the complete diff vs baseline (all records below included), not an increment.
- Build the engine (csrc changes: GDN kernels, Marlin tile, bindings).
- Three JIT CUDA extensions build at first server start into TORCH_EXTENSIONS_DIR, one directory per source hash (~40 s each, once): host_staged_ar.cu, dma_staged_ar.cu, lmh_codec.cu. Prebuild them by starting the server once, or by importing the modules.
- Serve with the task's flags plus `serve_args.txt`:
  - `--attention-backend TRITON_ATTN`
  - prefill CUDA graphs: a capture cap of 2048 tokens and an every-32 size grid
  - MTP k=2 with local-argmax reduction, FP8 draft lm_head and a 98,304-token draft vocab.
- No env vars needed: every optimization is on by default.

## What is in it (lineage; each step has its own bank record with evidence and validity review)
| record | official | change |
|---|---|---|
| T1-005 | 699.2 | host-staged (pinned host memory) all-reduce for TP=2 over PCIe; prefill CUDA graphs up to 2048 tokens |
| T1-009 | 730.9 | MTP k=2 spec decode; greedy fast path (sharded argmax + (max, idx) all-gather, greedy-id rejection sampler) |
| T1-012 | 749.7 | GDN replay kernel: committed fp32 state + log of accepted-token inputs, exact replay instead of per-token state stores; flush kernel materializes the stock layout when needed |
| T1-016 | 805.7 | FP8 draft lm_head over a 98,304-token subset (from non-benchmark text, ruling 2); Marlin {64,128,128} tile for small M; TRITON_ATTN |
| T1-019 / T1-020 | 837.0 / 850.3 | extra host-AR blocks L2-prefetch (evict_last) the next GEMM's weights (gate_up, in/out_proj, lm_head prefixes) |
| T1-023 | 861.5 | GDN replay also on mixed steps |
| T1-025 | 874.8 | hybrid AR for steps >= 256 tokens: 64-aligned token split, chunk 0 via copy-engine (DMA) AR, chunk 1 via host AR |
| T1-026 | 880.9 | workers self-bind to the GPU's NUMA node (numactl absent), so the pinned slots land near the GPU |
| T1-030 | 897.3 | lossless host-AR payload compression (block-exponent code, raw fallback per block; ratio 0.781) |
| T1-035 | 906.3 | lazy GDN checkpoint: committed state stored at most every 8 accepted tokens; replay from the log (bitwise) |
| T1-037 | 922.8 | target lm_head stored losslessly encoded (0.781, verified bitwise at load, raw freed) with a custom GEMM decoding in registers |
| T1-040 | 929.1 | RED8: GDN row dots reduced over 8 lanes instead of 32 (fp32 reorder only) |
| T1-041 | 934.1 | GDN replay scheduling: batched prologue + warp-private cp.async staging (bitwise) |
| T1-042 | 942.2 | copy-engine AR pipelined in 4 pieces with per-piece doorbells (bitwise) |

## Opt-out switches (defaults = the official configuration)
VLLM_DISABLE_HOST_STAGED_AR=0, VLLM_HOSTAR_COMP=1, VLLM_HOSTAR_L2PF=1 (BLOCKS 8, MAX_MB 88, STRIDE 32), VLLM_DMA_SPLIT=1 (MIN_TOKENS 256), VLLM_DMA_PIPE=4, VLLM_NUMA_SELF_BIND=1, VLLM_GDN_REPLAY=1, VLLM_GDN_MIXED_REPLAY=1, VLLM_GDN_LAZY=1 (CAP 8), VLLM_GDN_RED8=5, VLLM_LMH_CODEC=1.
Debug only: VLLM_GDN_REPLAY_STATS, VLLM_DMA_SPLIT_CHECK, VLLM_TEST_GREEDY_IDS_CHECK (all 0).
Note: VLLM_LMH_CODEC_PF defaults to "1" here (T1-037's prefetch of the encoded lm_head prefix). T1-038's v2 codec, which turned it off, was not adopted.

## Correctness summary
- Same math as baseline up to fp reordering:
  - bitwise changes: AR compression, GDN replay/lazy/scheduling, DMA pipelining, lm_head weight re-encoding;
  - reorders: RED8 row dots, the lm_head GEMM's summation order, AR summation (as baseline).
- Spec decode is exact (greedy verify).
- Precision is unchanged: FP8 KV cache, bf16 activations, bf16 AR payloads (losslessly compressed).
- No benchmark-specific logic: the draft vocab subset comes from non-benchmark text.
- Memory: lm_head raw shard freed (-0.28 GB/rank); the KV cache stays at 5 GiB (267,721 tokens); the 128K request passes.

## Requirements and assumptions
- 2 GPUs on one NUMA node without P2P (pinned host staging). The hybrid split, host AR and DMA AR assume TP=2.
- /dev/shm for the pinned slots.
- The JIT build needs nvcc at runtime.
