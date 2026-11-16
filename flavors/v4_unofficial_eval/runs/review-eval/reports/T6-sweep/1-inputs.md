# T6 sweep, step 1: the framework's inputs (leader-6, 2026-11-16 ~05:55 UTC)

## Model (config.json of Qwen/Qwen3.8-27B-FP8, text part)
- 64 layers: 48 Gated-DeltaNet (GDN, linear attention) + 16 full attention (every 4th, `full_attention_interval 4`).
- hidden d=5120, MLP m=17408 (SiLU-gated: gate_up 5120->2x17408, down 17408->5120).
- Full attention: 24 q heads, 4 KV heads, head_dim 256, attn output gate (q proj also produces a gate), partial RoPE (25% of dims), GQA 6:1.
- GDN: 16 key heads x128, 48 value heads x128, conv kernel 4; state S[r,l,h_v,128,128] in fp32 (`mamba_ssm_dtype float32`).
- vocab 248,320, untied lm_head (bf16, 2.54 GB total, 1.27 GB per GPU at TP=2).
- MTP: 1 extra layer, 2 draft tokens per step (serve args).
- Weights FP8 e4m3, 128x128 block scales; activation scheme dynamic (W8A8 per-token-group-128 allowed by ruling).

Per-GPU weight bytes per forward (TP=2): ~12.2 GB body (SPEED_OF_LIGHT), lm_head 1.27 GB, MTP layer ~0.2 GB + FP8 draft lm_head (0.64 GB per GPU at fp8).

## Hardware
2x RTX 4090 (sm_89), 24 GB, ~1.0 TB/s DRAM each, 72 MB L2, FP8 TC ~165 TFLOP/s (fp32 acc), bf16 TC ~165 (fp32 acc ~82 on GeForce? spec: 165 dense fp16 acc, 82.6 fp32 acc for bf16).
No P2P: GPU-GPU traffic goes through host memory over PCIe 4.0 x16 (~25 GB/s per direction). 48 CPU cores.

## Workload & objective (TASK.md)
- ShareGPT 400 prompts, ~260 input tokens, ~228 output tokens (reference length), temperature 0, closed loop c=16.
- Metric: output tok/s (mean of 2 fresh starts). Gates: TTFT p99 <= 1609 ms, TPOT p99 <= 55.3 ms, GSM8K-250 >= 89.6%, 128K request served.

## Contract (task_definition.md)
Same function up to fp reordering; FP8 weights and FP8 KV; activations bf16 or checkpoint W8A8 dyn-128 (fp32 cross-block accumulation); communication at baseline precision (bf16); exact spec decode only; prefix caching on; parsers on; max-num-seqs >= 16; nothing fitted to the benchmark; memory changes tested with the 128K request.

## Starting point: the champion T3-007 (853.6 tok/s), read from its code and serve args
- TP=2, max-num-seqs 16, FP8 KV cache, `--kv-cache-memory 4.4 GiB`, prefix caching on.
- MTP 2 drafts, FP8 draft lm_head, local-argmax reduction in the drafter; batch-sharded sampling (no full-logits all-gather).
- Linear layers: W8A8 Triton block-scaled GEMM (`fp8_block_w8a8=triton`) with tuned tile JSONs for all 5 linear shapes per GPU (qkvz 5120->?; gate_up N=17408 per GPU; down K=8704; o_proj K=3072; attn qkv N=7168/8192); activation quant fused into the preceding RMSNorm (`gemma_add_rms_norm_dyn128`) and into SiLU-and-mul (`fused_act_quant.py`).
- All-reduce: custom host-mapped pinned-memory all-reduce (hmar.cu, JIT) for <=96 rows (decode), NCCL above.
- L2 prefetch: a side-stream Triton kernel loads the first 16 MiB of the next GEMM's weights into L2 during each decode all-reduce.
- Attention: TRITON_ATTN backend (Q bf16, K/V fp8 dequantized in-kernel) for target and drafter, so the draft decode graph replays in place.
- CUDA graphs: decode (FULL) and mixed steps up to 2048 tokens (`max_prefill_cudagraph_capture_tokens 2048`).
- Measured (bank/T3-007/eval/1, job 75): 853.6 tok/s; trial1 duration 106.6 s for 91,079 tokens; acceptance length 2.34 (pos1 0.78, pos2 0.56); TPOT p50 17.2 ms; ITL median 31.1 ms (= one decode step, emitting ~2.34 tokens per request).
