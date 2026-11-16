# T7 sweep, step 1: inputs (leader-7)
Time budget note: T7 opened at 06:59 UTC with the run ending 08:15; files are compact on purpose.

## Model (config, task/SPEED_OF_LIGHT.md)
Qwen3.8-27B-FP8: 64 layers = 48 Gated-DeltaNet (GDN, linear attention, fp32 state 48 v-heads x 128 x 128 per layer/request)
+ 16 full-attention (every 4th; GQA, FP8 KV cache). hidden 5120, MLP 17408, vocab 248,320 (untied bf16 lm_head, 1.27 GB/GPU at TP=2).
FP8 128x128 block-scaled weights (~12.2 GB/GPU at TP=2). 1-layer MTP head (draft), reused recursively for k draft steps.

## Hardware
2x RTX 4090: ~1.0 TB/s DRAM, 24 GB, FP8 TC ~165 TFLOP/s (fp32 acc), no P2P: all-reduce via host memory over PCIe 4.0 (~25 GB/s/dir).
Ridge N ~ 80-160 rows (SoL) -> decode (M=48) is memory-bound.

## Workload / objective
ShareGPT 400 prompts, ~260 in / ~228 out, closed loop c=16, temperature 0. Score = output tok/s mean of 2 fresh starts.
Gates: TTFT p99 <= 1609 ms, TPOT p99 <= 55.3 ms, GSM8K-250 >= 89.6%, 128K request served.

## Contract
task_definition.md: same math up to fp reordering; FP8 weights + FP8 KV; bf16 or checkpoint W8A8 dyn-128 activations;
communication at baseline precision; exact spec decode only (method / draft count open); prefix caching on; nothing fitted to the benchmark.

## Starting point: champion T5-001 (895.4 tok/s), read from bank/T5-001/change.diff + serve_args.txt
- TP=2; MTP k=2 drafts, FP8 draft lm_head, local-argmax draft reduction, batch-sharded sampling.
- Linear layers: Triton W8A8 block-FP8 GEMMs with tuned configs for M in {1..96..}, fused SiLU+quant and norm+quant.
- All-reduce: custom host-mapped pinned-memory AR (HMAR) for <=96 rows, NCCL otherwise; L2 prefetch of next GEMM weights on a side stream during AR.
- Attention: TRITON_ATTN (bf16 Q, fp8 KV dequant in-kernel). KV budget 4.4 GiB.
- GDN: fused MTP decode kernel with DeltaLog (writes state after row 0 + a log of rows 1..T-1, kMaxLogRows=3, kMaxMtpTokens=8; replays accepted rows next step).
- CUDA graphs incl. prefill up to 2048 tokens.
- Measured (bank/T5-001/eval/1): acceptance length 2.343, per-position acceptance 0.782 / 0.567; TPOT p50 16.2 ms; TTFT p99 ~1030 ms; TPOT p99 35.6/38.9.
