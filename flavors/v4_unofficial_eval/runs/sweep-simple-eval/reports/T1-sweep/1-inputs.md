# T1 sweep, step 1: the framework's inputs for this task

Written in phase 1 (from framework/, task/, task_definition.md, assumption_ledger.md, focus.md,
champion.md and the BASE code only).

## Model (config.json of Qwen/Qwen3.8-27B-FP8, `text_config`)
- 64 layers: 48 Gated-DeltaNet (GDN, linear attention) + 16 full attention (every 4th layer,
  `full_attention_interval 4`). hidden d = 5120, MLP m = 17408 (SiLU-gated), vocab V = 248,320,
  untied embeddings / lm_head (bf16 lm_head: 248320 x 5120 = 2.54 GB, 1.27 GB per GPU at TP=2).
- Full attention: 24 query heads x head_dim 256, 4 KV heads (GQA 6:1), output gate
  (`attn_output_gate`: q_proj emits 2x), partial RoPE (25% of dims, mrope interleaved), qk-norm.
- GDN: 16 key heads x 128, 48 value heads x 128, conv kernel 4, recurrent state per layer per
  request 48 x 128 x 128 in **fp32** (`mamba_ssm_dtype float32`) = 3.1 MB (1.57 MB per GPU at TP=2).
- MTP: 1 extra layer (`mtp_num_hidden_layers 1`, full-attention layer + fc fusing embed/hidden),
  shares the embedding and lm_head with the target.
- FP8 checkpoint: 128x128 block-scaled e4m3 weights, `activation_scheme dynamic`.

Per-layer weights (FP8 bytes, whole model; per GPU at TP=2 = half):
| block | shape | params |
|---|---|---|
| GDN in_proj_qkvz | 5120 -> 2048+2048+6144+6144 = 16384 | 83.9 M |
| GDN in_proj_ba (bf16, unquantized) | 5120 -> 96 | 0.5 M |
| GDN out_proj | 6144 -> 5120 | 31.5 M |
| Attn qkv (q gated x2) | 5120 -> 12288+1024+1024 | 73.4 M |
| Attn o_proj | 6144 -> 5120 | 31.5 M |
| MLP gate_up | 5120 -> 2 x 17408 | 178.3 M |
| MLP down | 17408 -> 5120 | 89.1 M |
GDN layer 383 M, attention layer 372 M; body = 48 x 383 + 16 x 372 = 24.3 G params = **12.2 GB FP8
per GPU** (matches SPEED_OF_LIGHT). MTP layer ~0.42 GB (0.21 per GPU).

## Hardware (TASK.md, SPEED_OF_LIGHT.md; machine-specific)
| resource | value | note |
|---|---|---|
| F (bf16 / FP8 with fp32 accumulate) | ~165 TFLOP/s dense per GPU | FP8 with fp16 accumulate ~330 |
| BW (GDDR6X) | ~1.0 TB/s per GPU (1008 spec) | ncu: Marlin reaches 0.63-0.81 TB/s |
| Link_BW | no P2P; PCIe 4.0 x16 via host: ~25 GB/s per direction per GPU | GPU->host->GPU |
| C | 24 GB per GPU, 0.92 utilization | KV budget 5 GiB (fp8) |
| ridge N (FP8 weights, bf16 compute) | F/BW x b_W/2 = 165/1.0 x 1/2 ~ **82 rows** | not 333 (framework example) |

## Workload and objective
- ShareGPT, 400 prompts, ~260 input tokens (with chat template), ~228 output tokens (reference
  length), chat completions, temperature 0, closed loop at concurrency 16.
- Objective: output tok/s (BASE official 635.9). Gates: TTFT p99 / TPOT p99 within +20% of BASE
  (1341 / 46.1 ms), GSM8K-250 >= 89.6%, one 128K request served.
- Snapshot that dominates: 16 requests each with ~260..500 kept tokens, 3 ready rows each (1 + 2 MTP
  drafts) = M = 48 rows per decode step; plus, ~once per request, a mixed step with one ~215-260 token
  prompt chunk + ~45 decode rows.

## Contract (what counts as the same computation)
Same math up to fp reordering; FP8 weights + FP8 KV fixed; bf16 activations or the checkpoint's
W8A8 (dynamic per-token-group-128 FP8 activations, fp32 across K-blocks); communication at bf16
(no quantized all-reduce); exact speculative decoding only (method / draft count open); FP8 or
reduced-vocab (non-benchmark data) draft lm_head allowed; GDN/attention math, norms, KV, target
lm_head unchanged; nothing fitted to the benchmark; parsers on; max-num-seqs >= 16; 128K must fit.

## Starting point: the champion's design (BASE, empty diff; harness/serve_args.sh)
- TP=2 over both GPUs (split heads / MLP columns; 2 all-reduces per layer = 128 per target forward,
  + 2 per MTP step), vocab-parallel lm_head with an **all-gather of logits** before sampling.
- All-reduce backend: **PyNCCL only** (custom AR disabled: no P2P; symm-mem unsupported on sm_89;
  `FLASHINFER_PCIE_IPC` exists in this vLLM but is opt-in via `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`
  and off). NCCL picks `AllReduce_Sum_bf16_RING_LL` for every size.
- FP8 linear layers through Marlin (weight-only FP8, bf16 MMA, fp32 accumulate).
- KV: FP8, FlashInfer attention; GDN: fused recurrent decode kernel with MTP support
  (`gdn_decode_post_conv_mtp_kernel`), chunked kernels for prefill; fp32 state kept in place.
- MTP speculative decoding, 2 draft tokens, greedy verification.
- Scheduler: chunked prefill on, prefix caching on, max_num_scheduled_tokens 2048, max 16 seqs.
- Compilation: `CompilationMode.NONE` (no torch.compile / inductor fusions); CUDA graphs
  FULL_AND_PIECEWISE (decode graphs for <=16 requests, prefill graphs only up to 96 tokens, so
  real prompt chunks run eager).
