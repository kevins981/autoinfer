# T1 sweep, step 1: the framework's inputs for this task

Sources: model config.json (HF hub, Qwen/Qwen3.8-27B-FP8), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, task/harness/serve_args.sh, BASE server.log
(task/baseline/official/trial1/server.log), BASE code (bank/BASE: empty diff = vLLM 4527515cd7).

## 1. Model (config.json, text_config)
| | value |
|---|---|
| layers | 64 = 48 GDN (gated delta-net linear attention) + 16 full attention (every 4th) |
| hidden d | 5120; MLP intermediate m = 17408 (SiLU-gated, dense) |
| full attention | 24 q heads, 4 KV heads, head_dim 256, output gate (swish), partial RoPE 0.25, q/k norm |
| GDN | 16 key heads x 128, 48 value heads x 128, conv kernel 4, state 48 x 128 x 128 per layer per request, **fp32** (`mamba_ssm_dtype: float32`) |
| vocab | 248,320, untied embeddings; lm_head bf16 (in `modules_to_not_convert`) |
| MTP | 1 layer (`mtp_num_hidden_layers: 1`), shares embeddings and (in vLLM) the target lm_head |
| quantization | FP8 e4m3, 128x128 block weight scales, activation_scheme dynamic. Not converted: lm_head, embed, norms, GDN in_proj_a/b(a), conv1d, A_log, dt_bias |

Per GPU at TP=2 (napkin): body FP8 weights ~12.5 GB read per forward (ncu: 4 GEMMs per layer read
43 + 16 + 91 + 46 MB = 196 MB; x64 = 12.5 GB); lm_head 124,160 x 5120 bf16 = 1.27 GB; MTP layer ~0.2 GB.

## 2. Hardware (TASK.md, SPEED_OF_LIGHT.md)
| | per GPU |
|---|---|
| DRAM BW | ~1.0 TB/s spec; Marlin achieves ~0.63-0.80 TB/s (ncu key csv), cutlass lm_head GEMM ~0.93 TB/s (nsys: 1.27 GB in 1.36 ms) |
| bf16 tensor (fp32 acc) | ~165 TFLOP/s dense (Marlin computes in bf16) |
| FP8 tensor | 165 (fp32 acc) / 330 (fp16 acc) TFLOP/s per SPEED_OF_LIGHT (marked "check"; the Ada whitepaper may list FP8/fp32-acc higher, to verify by microbenchmark) |
| link | no P2P; GPU-GPU through host memory, PCIe 4.0 x16 ~25 GB/s per direction |
| capacity | 24 GB; gpu-memory-utilization 0.92; KV budget 5 GiB (fp8 KV + GDN states) |
| ridge (weight-only FP8, bf16 compute) | N_ridge = b_W F / (2 BW) = 1 x 165e12 / (2 x 1e12) ~ 82 rows per weight load (vs 333 in the framework's example) |

## 3. Workload and objective
- vllm bench serve, ShareGPT, 400 prompts, ~261 input tokens (104,526 total), ~228 output tokens
  (91,080 total), temperature 0, closed loop at concurrency 16. Score = output tok/s (mean of 2 fresh
  starts). BASE: 635.9 tok/s, benchmark duration 143.2 s per trial.
- Gates: TTFT p99 <= 1.2 x 1341 ms, TPOT p99 <= 1.2 x 46.1 ms; GSM8K-250 >= 89.6%; one 128K request.
- Snapshot in steady state: 16 requests, each with ~260 + (0..450) known tokens; kept: FP8 KV for
  16 attention layers, fp32 GDN state + conv state for 48 layers; per step 3 ready rows per request
  (1 verified + 2 MTP drafts), i.e. M ~ 42-48 rows (nsys median 42). New requests arrive one at a
  time (closed loop) -> a mixed step with ~215 prefill tokens + ~45 decode rows (nsys median).

## 4. Contract (what counts as the same computation)
- Same function up to fp reordering; FP8 weights, FP8 KV; bf16 activations or the checkpoint's own
  W8A8 (dynamic per-token-group-128 FP8 activations, fp32 accumulation across K-blocks); comm,
  norms, attention/GDN math, KV, target lm_head stay at baseline precision.
- Exact speculative decoding only. Draft side is free: FP8 draft lm_head allowed; reduced-vocab
  draft lm_head allowed (vocab chosen from non-benchmark text); trained draft heads allowed (non-
  benchmark data).
- Prefix caching stays on; parsers on; max-num-seqs >= 16; 128K request must fit.
- Nothing fitted to the benchmark.

## 5. Starting point: the champion's design (BASE, read from server.log + code)
- TP=2 over both GPUs (all weights split by output/input channel, 2 all-reduces per layer).
- All-reduce: **PYNCCL only** (custom all-reduce disabled: no P2P; the opt-in FlashInfer PCIe-IPC
  all-reduce `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` is available in this build but off). NCCL
  kernel: `AllReduce_Sum_bf16_RING_LL`, 2 channels (grid 2x1x1).
- Linear layers: Marlin weight-only FP8 (bf16 compute), "custom fusions: norm_quant, act_quant".
- Full attention: FlashInfer (fp8 KV, bf16 query); GDN: Triton/FLA chunked prefill kernels, a custom
  `gdn_decode_post_conv_mtp_kernel<float,3>` for decode with MTP.
- V2 model runner; CUDA graphs FULL_AND_PIECEWISE, capture sizes up to 96 tokens; decode steps in full
  graphs; prefill steps (>96 tokens) eager-ish (piecewise not used for >96: compile mode NONE).
- Chunked prefill on, max_num_scheduled_tokens 2048, prefix caching on.
- MTP spec decode, 2 draft tokens, method mtp, draft uses the target's bf16 lm_head (full vocab);
  acceptance length 2.345 (per position 0.78, 0.56).
- Sampling: FlashInfer top-k/top-p sampler; vocab-parallel lm_head -> logits all-gather.
