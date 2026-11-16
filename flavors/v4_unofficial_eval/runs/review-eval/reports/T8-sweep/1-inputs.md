# T8 sweep 1: inputs (leader-8, 2026-11-16 ~07:10 UTC; written under a 70-minute run deadline, so compact)

## Model (config.json via SPEED_OF_LIGHT.md inputs; Qwen/Qwen3.8-27B-FP8)
- 64 layers: 48 GDN (gated delta-net linear attention, fp32 state 128x128 per v-head) + 16 full-attention (every 4th), hidden 5120, MLP 17408 (gate_up fused), vocab 248,320 untied bf16 lm_head.
- FP8 128x128 block-scaled weights; MTP draft head (1 layer + output layer), 2 draft tokens.
## Hardware
- 2x RTX 4090 (sm_89), 24 GB, ~1.0 TB/s DRAM, ~165 TFLOP/s FP8 (fp32 acc), 72 MB L2, no P2P: link = host memory over PCIe 4.0 (~25 GB/s/dir).
## Workload / objective
- 400 ShareGPT prompts (~260 in, ~228 out), closed loop c=16, temperature 0; output tok/s, mean of 2 fresh starts; TTFT/TPOT p99 guard (<=1.2x BASE), GSM8K-250 >= 89.6%, 128K request.
## Contract
- task_definition.md: same function up to fp reordering; FP8 weights + FP8 KV; bf16 or the checkpoint's dyn per-token-group-128 W8A8 activations; exact speculative verification; prefix caching on; parsers on; nothing fitted to the benchmark.
## Starting point: champion T5-001 (895.4 tok/s), as read from bank/T5-001 (change.diff, serve_args.txt, record)
- TP=2; MTP k=2 with fp8 draft lm_head, local argmax, batch-sharded sampling.
- Linear layers: Triton W8A8 block FP8 (tuned configs per M), fused act/norm quant.
- All-reduce: custom host-mapped (HMAR) for <=96 rows, NCCL above; side-stream L2 prefetch of the next GEMM's weights during each AR.
- Attention: TRITON_ATTN (bf16 Q, fp8 KV dequant in kernel); KV budget 4.4 GiB (235,650 tokens).
- GDN: fused MTP decode kernel with DeltaLog (1 state write + log instead of 3 state writes); materialize before non-decode consumers.
- CUDA graphs: FULL decode graphs to 16 req, piecewise for mixed steps up to 2048 tokens.
- Measured (bench_c16.json, eval/1 trial1): 101.3 s for 91,036 output tokens; acceptance length 2.35 (pos0 78.3%, pos1 56.8%); TPOT p50 16.2 ms, ITL p50 28.7 ms.
