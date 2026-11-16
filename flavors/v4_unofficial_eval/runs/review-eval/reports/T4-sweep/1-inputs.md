# T4 sweep, step 1: the framework's inputs for this task

Sources: config.json of Qwen/Qwen3.8-27B-FP8 (hf_hub_download), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, champion.md, bank/T3-001 (change.diff, serve_args.txt,
eval/1 server.log + bench.log), workspace t4-champ (= T3-001).

## 1. Model (text model only; `--language-model-only`)
| item | value | per GPU at TP=2 |
|---|---|---|
| layers | 64 = 48 GDN (linear attention, "Gated DeltaNet") + 16 full attention (every 4th) | all layers on both GPUs |
| hidden d | 5120 | replicated residual stream |
| MLP | SiLU-gated, intermediate 17408 | gate_up N=17408 (2x8704), down K=8704 |
| full attention | 24 q heads, 4 kv heads, head_dim 256, output gate (q proj doubled), partial RoPE 0.25 (64 dims), mRoPE | qkv N=7168 (12 q-heads x 256 x 2 + 2 kv x 256 x 2), o_proj K=3072; 2 kv heads/GPU |
| GDN | 16 key heads x 128, 48 value heads x 128, conv kernel 4, state fp32 (`mamba_ssm_dtype`) | in_proj_qkvz N=8192 (q 1024, k 1024, v 3072, z 3072), in_proj_ba N=48 (bf16, small), conv on 5120 channels, out_proj K=3072; 8 k-heads / 24 v-heads per GPU |
| GDN state | per layer per request: 24 heads x 128 x 128 fp32 = 1.57 MB/GPU | x48 layers x16 req = 1.21 GB/GPU |
| vocab | 248,320, untied; lm_head bf16 (in `modules_to_not_convert`) | 124,160 x 5120 bf16 = 1.27 GB/GPU |
| MTP draft | 1 full-attention decoder layer (FP8, ~0.19 GB/GPU) + fc (2d -> d) + its own lm_head (ParallelLMHead, loaded from the checkpoint's lm_head, bf16) | run twice per decode step (2 draft tokens) |
| weights | FP8 e4m3, 128x128 block scales, activation_scheme dynamic | body ~12.2 GB/GPU (48x191 MB + 16x186 MB) |

Per-layer FP8 GEMM bytes per GPU: qkvz 41.9 MB, out_proj/o_proj 15.7 MB, attn qkv 36.7 MB, gate_up 89.1 MB, down 44.6 MB.

## 2. Hardware
2x RTX 4090 (sm_89, 24 GB GDDR6X, ~1.0 TB/s each, 72 MB L2), FP8 tensor cores ~165 TFLOP/s with fp32
accumulate (330 fp16-acc; the task's accumulation rule applies across K-blocks), bf16 ~165 TFLOP/s dense.
No P2P: GPU<->GPU through host memory over PCIe 4.0 x16 (~25 GB/s per direction, several us latency per
hop). 48 CPU cores, 125 GB RAM. Ridge point for FP8 weights (b_W=1) at fp32-acc: N_ridge = 165e12/(2*1e12) ~ 80 rows;
bf16 weights (lm_head): ~165 rows.

## 3. Workload and objective
`vllm bench serve`, ShareGPT 400 prompts (mean ~261 input tokens with template, output = reference length,
mean ~227), chat completions, temperature 0, closed loop at concurrency 16; score = output tok/s, mean of 2
fresh server starts. Champion run (eval/1 trial 1): 111.8 s, 104,526 input tokens, 90,684 output tokens.
Workload snapshot for the framework (steady state): ~15-16 running requests, each with ~260-500 tokens kept
(KV for 16 attention layers + one GDN state per GDN layer), 3 ready rows per request per decode step
(1 verified token + 2 MTP drafts: M = 48 rows at the target's GEMMs); a new request arrives every ~0.28 s
(400 in 112 s) and brings ~261 prompt rows in a mixed step. Latency gates: TTFT p99 <= 1609 ms, TPOT p99
<= 55.3 ms (1.2x BASE).

## 4. Contract (what counts as the same computation)
task_definition.md: same function up to fp reordering; FP8 weights, FP8 KV; activations bf16 or the
checkpoint's W8A8 dynamic per-token-group-128 (fp32 accumulation across K-blocks); communication at
baseline precision (bf16 all-reduce, fp32 add of 2 partials); norms/attention/GDN/KV/target lm_head
unchanged; exact speculative decoding only (method and draft count may change); draft-side changes free
(FP8 draft lm_head ruled OK; reduced-vocab draft OK with non-benchmark frequencies; trained heads on
non-benchmark data OK); prefix caching on; parsers on; max-num-seqs >= 16; 128K request must work; nothing
fitted to the benchmark. Promotion: mean of starts higher and every start above the champion's mean (<1%
gain needs a 3rd start).

## 5. Starting point: the champion T3-001 (816.6 tok/s), as read from its code and logs
- **Parallelism**: TP=2 over all layers (heads / MLP channels split; 2 all-reduces per layer = 128 per
  target forward + the MTP layer's), lm_head vocab-split.
- **GEMMs**: W8A8 Triton block-scaled FP8 kernel (`linear_backend_per_quant fp8_block_w8a8=triton`) with
  tuned tile JSONs for the 5 shapes; activation quant fused into the producers (Gemma add-RMSNorm + dyn128
  quant Triton kernel; SiLU-mul + per-block quant CUDA kernel).
- **All-reduce**: HMAR (custom host-mapped pinned-memory all-reduce, 16 blocks x 512 threads, flag spin,
  fp32 add) for <= 96 rows (decode), PyNCCL ring for bigger (mixed/prefill steps). The all-reduce before
  each norm is an explicit call followed by the fused norm+quant kernel (not fused into the AR).
- **L2 prefetch** (T3): during each hmar AR, a 32-program side-stream Triton kernel reads the first 16 MiB of
  the next GEMM's weight (qkvz/qkv or gate_up) with evict_last.
- **Speculation**: MTP, 2 draft tokens, chain; draft uses local argmax (no logits all-gather);
  batch-sharded sampling; acceptance ~2.33 tokens/step (per-position 0.775 / 0.558). Draft lm_head is the
  bf16 one (T1-001's FP8 draft lm_head flag is NOT in the champion's serve args).
- **CUDA graphs**: FULL_AND_PIECEWISE, breakable graphs; decode FULL graphs for 1..16 requests;
  prefill/mixed piecewise graphs up to 2048 tokens (max_prefill_cudagraph_capture_tokens=2048); the MTP
  drafter is captured ("autoregressive speculator"), but FlashInfer cannot update draft metadata in place:
  draft attention metadata rebuilt eagerly.
- **Attention**: FlashInfer (bf16 query, fp8 KV), GDN spec-decode kernel "cuda"
  (`gdn_decode_post_conv_mtp_kernel`); chunked prefill on, max_num_scheduled_tokens 2048; prefix caching on.
- **Memory**: model 14.22 GiB/GPU, KV 5 GiB (267,721 tokens), block size 1600 (hybrid lcm).
- **Measured** (eval/1): 816.6 tok/s (811.3/821.8), median ITL 32.4 ms (= one decode step, 2.33 tokens per
  request), median TPOT 18.0 ms, median TTFT 171 ms, acceptance 2.333.
