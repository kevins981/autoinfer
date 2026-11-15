# T3 sweep, step 1: the framework's inputs for this task

Sources: model `config.json` (HF hub), task/TASK.md, task/SPEED_OF_LIGHT.md, task_definition.md,
assumption_ledger.md, HEAD = T1-016 (`bank/T1-016/change.diff`, `serve_args.txt`, its unofficial
results `bank/T1-016/unofficial/1/`). Read in phase 1 only (no other team's files).

## 1. Model (Qwen3.8-27B-FP8, `qwen3_5_text`)
| item | value | derived |
|---|---|---|
| layers L | 64 = 48 GDN (linear attention) + 16 full attention (every 4th: l = 3,7,...,63) | |
| hidden d | 5120 | residual row = 10 KB bf16 |
| MLP | dense SwiGLU, m = 17408 (no MoE) | gate_up 5120x34816, down 17408x5120 |
| full attention | 24 Q heads, 4 KV heads (GQA 6:1), head_dim 256, attn_output_gate (q proj also emits a gate), partial rotary 0.25 (64 of 256 dims) | q+gate 12288, k 1024, v 1024 -> qkv proj 5120x14336; o_proj 6144x5120 |
| GDN | 16 key heads x 128 (q,k), 48 value heads x 128 (v, z gate), conv1d kernel 4, ssm state fp32 (`mamba_ssm_dtype`) | in_proj 5120x(2048+2048+6144+6144 + 48+48) = 5120x16480; out_proj 6144x5120 |
| GDN state S | per layer per request: 48 x 128 x 128 fp32 = 3.15 MB (1.57 MB per GPU at TP=2) | x48 layers = 75.5 MB/request/GPU; x16 = 1.21 GB/GPU |
| conv state | 3 x (2048+2048+6144) bf16 per layer per request = ~61 KB (30 KB/GPU) | small |
| vocab V | 248,320, untied lm_head bf16 | 2.54 GB total, 1.27 GB per GPU (vocab-sharded) |
| MTP | 1 extra layer (attention-type + MLP, FP8) + shares the target lm_head/embedding | |
| quantization | FP8 e4m3, 128x128 block weight scales, activation_scheme dynamic | |

FP8 weight bytes per layer (whole model; per GPU = half at TP=2):
- MLP 267 MB (178 + 89); GDN layer 116 MB (84.4 in_proj + 31.5 out_proj); attention layer 105 MB (73.4 qkv + 31.5 o).
- Body: 64 x 267 + 48 x 116 + 16 x 105 = 17.1 + 5.56 + 1.68 = **24.3 GB, 12.2 GB per GPU**.
- MTP layer ~0.4 GB (0.2/GPU); target lm_head 1.27 GB/GPU (bf16).
- FLOPs per token (body, whole model): 2 x 24.3e9 = ~48.6 GFLOP (24.3 per GPU), + lm_head 2 x 1.27e9 = 2.5 GFLOP.

## 2. Hardware
2 x RTX 4090 (sm_89, 128 SMs, 24 GB, ~72 MB L2), per GPU: DRAM ~1008 GB/s spec (napkin: ~0.9 TB/s
achievable for streaming), bf16 tensor ~165 TFLOP/s dense with fp32 accumulate, FP8 ~165 (fp32
accumulate) / ~330 (fp16 accumulate) per SPEED_OF_LIGHT.md. **No P2P**: GPU-GPU only through host
memory over PCIe 4.0 x16 (~25 GB/s per direction), so every TP all-reduce has a few-us latency
floor (two PCIe crossings + flag polling). 48 CPU cores (the CPU runs the scheduler and the
launch path). Ridge point N_ridge (FP8 weight, bf16 activations through Marlin) = b_W*F/(2*BW) ~
1*165e12/(2*0.9e12) ~ **90 rows** per weight load: decode at M=48 rows is memory-bound, prefill
chunks (hundreds of rows) are compute-bound.

## 3. Workload and objective
- Official: ShareGPT, 400 prompts, input ~260 tokens (incl. chat template), output = reference reply
  length (~228 mean), closed loop at concurrency 16, temperature 0. Score = output tok/s (mean of 2
  fresh-server trials). Unofficial eval: 200 prompts, 2 fresh starts vs HEAD's cached starts.
- Snapshot that dominates: 16 requests, each with ~260-500 tokens kept (KV for 16 attention layers;
  S for 48 GDN layers), 1 + 2 draft rows each -> **M = 48 rows per decode step**. Prefill: one
  request (~260 tokens) enters whenever one finishes: ~400 prefills, usually riding a decode step
  (chunked/mixed steps).
- Gates: GSM8K-250 >= 89.6; p99 TTFT / TPOT within +20% of BASE; 128K request must work.

## 4. Contract (what counts as the same computation)
Exact math up to fp reordering; FP8 weights + FP8 KV fixed; activations bf16 or the checkpoint's own
W8A8 (dynamic per-token-group-128 FP8 activations, fp32 cross-block accumulation), communication at
baseline precision; exact spec decode only (draft side may be approximate: fp8 draft lm_head and a
reduced-vocab draft argmax are ruled allowed); nothing fitted to the benchmark; prefix caching stays
on; max-num-seqs >= 16; parsers on.

## 5. Starting point: the champion's design (HEAD = T1-016, read from its diff + serve args)
Measured (T1-016 unofficial #1, 200 prompts, 2 starts): **808.9 / 808.7 tok/s**, MTP acceptance
length **2.33** (per position 0.78, 0.56), median ITL 29.0 ms (= one decode step), median TPOT
17.5 ms, p99 TTFT 1.21 s, p99 TPOT 48.7 ms; 46,045 output tokens in 56.9 s; 19,699 request-steps.
Official chain: CONFIRMED T2-018 838.0 tok/s (400 prompts), HEAD expected ~841.

Design, node by node:
| part | HEAD's choice |
|---|---|
| parallelism | TP=2 (every layer split on heads / m; 2 all-reduces per layer = 128 per forward + embedding/lm_head gathers) |
| all-reduce | **host pinned-memory one-shot all-reduce** (`csrc/host_all_reduce.cu`) for tensors <= 128 x 5120 bf16 (decode); NCCL for larger; for prefill rows >= 192 an eager **chunked GEMM + AR pipeline** (2 chunks, AR of chunk k on a side stream under GEMM of chunk k+1) |
| FP8 GEMMs | Marlin W8A16 (bf16 activations, weights repacked), with a mid-batch thread-config table for M<=48; **Triton W8A16** for GDN out_proj and attention o_proj (checkpoint layout) |
| GDN decode | fused post-conv MTP decode kernel; **replay mode** (ReplaySSM-like: keep S + pending k/v rows, one state write per step, lazy commit of up to 2 steps' rows) |
| full attention | FlashInfer decode (spec, q_len 3) with device-exact lengths planned from CPU upper bounds (no blocking .cpu()), split-KV kept |
| speculation | MTP, 2 drafts, chain; **fp8 draft lm_head**, draft argmax over token ids < 98,304 only (rows broadcast once at load), local argmax reduction (no full-vocab gather), batch-sharded sampling |
| graphs | CUDA graphs for decode; prefill/mixed steps graphed up to 512 tokens (`max_prefill_cudagraph_capture_tokens`) |
| KV | FP8, 5 GiB budget, block size per vLLM default, prefix caching on |
| other | async scheduling (vLLM v1 default, the model runner `gpu/model_runner.py`) |
