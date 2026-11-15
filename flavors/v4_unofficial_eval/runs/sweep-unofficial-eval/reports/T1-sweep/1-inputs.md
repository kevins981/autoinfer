# T1 sweep, step 1: the framework's inputs for this task

Sources: Qwen/Qwen3.8-27B-FP8 config.json (hf_hub_download), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, champion.md (HEAD = BASE), task/harness/serve_args.sh,
BASE server logs (task/baseline/official/trial1/server.log, task/baseline/nsys_c16/server.log).

## 1. Model (config: model_type qwen3_5_text)
| item | value |
|---|---|
| layers L | 64: 48 Gated-DeltaNet (GDN, linear attention) + 16 full attention (every 4th, `full_attention_interval 4`) |
| hidden d | 5120; MLP m = 17408 (SiLU gated: gate+up then down) |
| full attention | 24 q heads, 4 KV heads (GQA 6:1), head_dim 256, output gate (`attn_output_gate`), partial RoPE 0.25 (64 of 256 dims), qk-norm |
| GDN | 16 key heads x 128, 48 value heads x 128, conv kernel 4, state per head 128x128, **state dtype fp32** (`mamba_ssm_dtype float32`) |
| vocab V | 248,320, untied lm_head (bf16, 2.54 GB total) and embedding (bf16) |
| MTP | 1 extra layer (`mtp_num_hidden_layers 1`, full-attention block + fc 2d->d), shares embedding and lm_head with the target |
| weights | FP8 e4m3, 128x128 block scales (`weight_block_size [128,128]`, activation_scheme dynamic) |

Per-layer weight bytes (FP8, full model; per GPU at TP=2 is half):
- GDN layer: in_proj (q,k: 2x2048, v: 6144, z: 6144, b/a: 96 -> 16480 x 5120) 84.4 MB; out_proj 6144x5120 31.5 MB; MLP 3x5120x17408 = 267.4 MB. **383 MB**
- attention layer: qkv+gate (12288+2048) x 5120 = 73.4 MB; o_proj 31.5 MB; MLP 267.4 MB. **372 MB**
- body: 48x383 + 16x372 = 24.3 GB -> **12.2 GB per GPU** (matches SPEED_OF_LIGHT)
- lm_head bf16: 2.54 GB -> **1.27 GB per GPU** (vocab-parallel); MTP layer ~0.2 GB per GPU (+ the shared lm_head)

## 2. Hardware (per device)
| | value |
|---|---|
| devices | 2x RTX 4090 (sm_89), 24 GB each, 128 SMs |
| HBM (GDDR6X) BW | 1.008 TB/s spec; measured Marlin at M=48 reaches ~0.80 TB/s (ncu, baseline/ncu_marlin) |
| F bf16 (fp32 acc) | ~165 TFLOP/s dense; FP8 with fp32 acc ~165, FP8 with fp16 acc ~330 |
| link | **no P2P**: GPU<->GPU through host memory over PCIe 4.0 x16, ~25 GB/s per direction (spec; to be measured) |
| ridge N (Marlin: fp8 weights, bf16 compute) | F/(2*BW/b_W) = 165e12*1/(2*1e12) ~ **82 rows** per weight load (not 330: the framework's 4090 example assumes FP8 compute at 660 TF) |
| ridge for W8A8 FP8 (fp16 inner acc) | ~165 rows |
| host | 48 cores, 125 GB RAM, shared server |

## 3. Workload and objective
- Official: `vllm bench serve`, ShareGPT 400 prompts, input mean ~260 tokens (incl. chat template), output = reference length, mean ~228, closed loop, 16 concurrent, temperature 0, seed 0. Score = output tok/s (mean of 2 fresh-server trials).
- Snapshot at steady state: 16 requests, each with k,v (16 attention layers) and GDN state S kept, ~250-500 known tokens; 1 ready position + 2 MTP drafts each (M = 48 rows/step), plus every few steps one new request whose ~260 prompt rows are merged into the step (chunked prefill, max_num_scheduled_tokens 2048).
- Gates: latency (TTFT p99, TPOT p99 within +20% of BASE official), GSM8K-250 >= 89.6%, one 128K request must work.

## 4. Contract (what counts as the same computation)
- Exact math up to fp reordering; FP8 weights and FP8 KV fixed; bf16 activations or the checkpoint's own W8A8 (dynamic per-token-group-128 FP8 activations, fp32 across K-blocks; needs a same-job GSM8K-250 A/B within 2 points).
- Communication at baseline precision (bf16 all-reduce; no compression).
- Speculation must be exact (greedy output identical to target). Draft-side changes are free as long as verification is exact: FP8 draft lm_head allowed, reduced-vocab draft head allowed with non-benchmark frequency data, retrained draft head allowed with non-benchmark data.
- GDN state dtype fp32 is part of the model's numerics (`mamba_ssm_dtype`): storing it at lower precision would be an approximation -> `approx` rows, not exact.
- Nothing fitted to the benchmark; parsers stay on; max-num-seqs >= 16; 128K request must fit.

## 5. Starting point: the champion's design (HEAD = BASE)
From serve_args.sh and the server log:
- **Split/Place**: TP=2 on every linear layer (heads/columns split, row-parallel o_proj/out_proj/down_proj followed by an all-reduce), vocab-parallel embedding and lm_head (logits all-gathered). All-reduce backend: **PYNCCL only** (custom all-reduce disabled: no P2P); kernels are `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` at every size.
- **Keep**: FP8 KV cache with a fixed 5 GiB budget (267,721 tokens), GDN state fp32 in the same block pool; prefix caching on (mamba "align" mode, precopy/postprocess kernels each step); 0 preemptions in the official run.
- **Rewrite**: weight-only FP8 via Marlin (bf16 activations, fp8 weights dequantized in-kernel); lm_head bf16 (cutlass/cublas); GDN decode via a fused `gdn_decode_post_conv_mtp_kernel`; flashinfer attention.
- **Speculate**: MTP, 2 draft tokens, greedy; acceptance length 2.345 (per position 0.78, 0.56) officially. The drafter's first step runs its layer on all 48 rows, then lm_head on 16 rows; each draft step all-gathers full draft logits (no local-argmax).
- **Window**: continuous batching, chunked prefill (prompt rows merged with decode rows), FULL_AND_PIECEWISE CUDA graphs (captures up to 96 tokens), torch.compile mode NONE (no inductor fusion), `fuse_allreduce_rms` off.
- Measured: official 635.9 tok/s; TPOT median 23.0 ms, ITL median 39.5 ms.
