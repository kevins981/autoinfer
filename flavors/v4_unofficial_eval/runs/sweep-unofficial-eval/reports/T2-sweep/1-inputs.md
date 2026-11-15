# T2 sweep, step 1: the framework's inputs for this task

Sources: `config.json` of Qwen/Qwen3.8-27B-FP8 (hf hub), task/TASK.md, task_definition.md,
assumption_ledger.md, task/SPEED_OF_LIGHT.md, the BASE server log, BASE profiles
(task/baseline/nsys_c16, ncu_marlin) and the engine code at HEAD (= BASE, empty diff).

## 1. Model (per scalar index; TP=2 halves shown as "per GPU")
| Item | Value |
|---|---|
| Layers `l` | 64: 48 Gated-DeltaNet (GDN, linear attention) + 16 full attention (every 4th: l = 3,7,...,63) |
| hidden `d` | 5120; RMSNorm (fused add+norm) before each mixer and MLP |
| MLP `m` | 17408 SiLU-gated (gate_up 5120→2x17408, down 17408→5120) |
| Full attention | 24 q heads, 4 kv heads (GQA 6:1), head_dim 256, output gate (q proj is 2x wide: q and gate), partial RoPE (64 of 256 dims, mrope interleaved), q/k RMSNorm |
| GDN | 16 key heads x 128, 48 value heads x 128 (3 v heads per k head), conv1d kernel 4 over q,k,v, gates a/b (in_proj_ba, bf16, unquantized), output gate z (swish) + RMSNorm, recurrent state S[r,l,h] = 128x128 **fp32** (`mamba_ssm_dtype: float32`) |
| Vocab | 248,320, untied embed and lm_head (bf16, not FP8) |
| MTP | 1 extra layer (`mtp_num_hidden_layers: 1`), shares embed + lm_head with the target, run 2 times per step (2 draft tokens) |
| Weights | FP8 e4m3, 128x128 block scales (`weight_block_size [128,128]`, `activation_scheme dynamic`) |

Per-GPU weight bytes (TP=2, FP8):
| Layer type | qkvz / qkv | out / o | gate_up | down | total |
|---|---|---|---|---|---|
| GDN (x48) | 8192x5120 = 41.9 MB | 3072x5120 = 15.7 MB | 17408x5120 = 89.1 MB | 8704x5120 = 44.6 MB | 191.3 MB (+ ba 0.5 MB bf16) |
| full attn (x16) | 7168x5120 = 36.7 MB | 15.7 MB | 89.1 MB | 44.6 MB | 186.1 MB |
| body | | | | | **12.16 GB** |
| lm_head (bf16, vocab-parallel) | 124,160 x 5120 x 2 B | | | | **1.27 GB** |
| MTP layer | ~0.2 GB (attn + MLP + fc) | | | | |

Per-request state per GPU: GDN state 48 layers x 24 v-heads x 128x128 x 4 B = **75.5 MB**;
full-attn KV fp8 = 16 layers x 2 kv heads x 256 x 2 x 1 B = **16 KB/token**.

## 2. Hardware (per GPU, x2)
| Resource | Value | Note |
|---|---|---|
| HBM BW | 1008 GB/s spec; Marlin reaches 0.63-0.80 TB/s (ncu: 82% "memory throughput" peak) | measured below in 3-calibration |
| Tensor compute | bf16 (fp32 acc) ~165 TFLOP/s dense; FP8 (fp32 acc) ~165, FP8 (fp16 acc) ~330 | Marlin computes in bf16 |
| Ridge N (FP8 weight bytes, bf16 compute) | N_ridge = b_W*F/(2*BW) = 1*165e12/(2*1e12) ≈ **82 rows** | so a decode step with M=48 rows is memory-bound, a mixed step with M≈300 is compute-bound |
| L2 | 72 MB | relevant for a prefetch idea (4-knobs) |
| Link | no P2P; NCCL ring through host memory (SHM), PCIe 4.0 x16 ~25 GB/s/dir spec | custom all-reduce disabled (no P2P) -> PyNCCL only |
| Capacity | 24 GB; weights 12.2 + lm_head 1.27 + embed 1.27 + MTP 0.2 ≈ 15 GB; KV budget 5 GiB (fixed by `--kv-cache-memory`) holds attention KV + GDN states | |

## 3. Workload and objective
- 400 ShareGPT prompts, chat API, input mean ~260 tokens, output = reference length (mean ~228),
  temperature 0, seed 0, closed loop at **concurrency 16**; score = output tok/s (mean of 2 fresh-server trials).
- Snapshot in steady state: 16 requests with kept KV/GDN state, each with 1 known-but-unprocessed
  token + 2 MTP drafts -> **M = 48 rows** per decode step; ~every 7th step also contains one or
  more new prompts (P ≈ 60-1900 prefill rows) -> mixed steps.
- MTP acceptance BASE: 2.345 tokens/step per request (per position 0.78, 0.56).
- Gates: TTFT p99 / TPOT p99 within +20% of BASE (1341 / 46.1 ms), GSM8K-250 >= 89.6%, one 128K request.

## 4. Contract (what counts as the same computation)
- Exact math up to fp reordering; FP8 weights, FP8 KV, bf16 activations or the checkpoint's own
  W8A8 (dynamic per-token-group-128 activations, fp32 accumulation across K-blocks; needs a same-job
  GSM8K-250 A/B within 2 points).
- Communication at baseline precision (bf16 all-reduce), norms/attention/GDN/KV/target lm_head unchanged.
- Spec decode exact (greedy output identical to target). Draft side free: FP8 draft lm_head ALLOWED,
  reduced-vocab draft lm_head ALLOWED (vocab from non-benchmark data), retrained draft head allowed
  (non-benchmark data).
- Nothing fitted to the benchmark; parsers on; max-num-seqs >= 16; 128K request must fit.

## 5. Starting point: the champion's design (HEAD = BASE, read from code + server log)
- vLLM `4527515cd7`, **Model Runner V2** (`vllm/v1/worker/gpu/`), MTP speculator (`TargetDependentAR`)
  with 2 draft tokens, greedy draft (`draft_sample_method=greedy`).
- TP=2 everywhere (attention heads, GDN heads, MLP inner dim, vocab-parallel embed + lm_head);
  2 all-reduces per layer (128 per forward) through PyNCCL (`AllReduce_Sum_bf16_RING_LL`, 2 CTAs).
- Linear layers: `MarlinFP8ScaledMMLinearKernel` (weight-only FP8, bf16 compute), chunks M in
  64-row blocks (thread_m_blocks <= 4) -> re-reads weights per 64 rows.
- GDN decode: `gdn_decode_post_conv_mtp_kernel` (fp32 state); GDN prefill: FLA chunked kernels.
- Attention: FlashInfer, FP8 KV, LBNHC layout.
- CUDA graphs: breakable CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=1` auto), decode steps replayed.
- Scheduler: `max_num_scheduled_tokens` 2048 (auto from spec settings), prefix caching on (0% hits).
- Target logits: lm_head on all 48 rows (bf16 cuBLAS/cutlass), then **all-gather of full logits**
  (48 x 248,320 bf16 ≈ 24 MB) before rejection sampling.
- Draft: per draft step, MTP layer + lm_head (bf16, full vocab) + all-gather of full logits + argmax.
- Engine options present but off: `draft_lm_head_quantization` (fp8), `use_local_argmax_reduction`,
  `enable_batch_sharded_sampling` (Qwen3.5 implements `compute_logits_local`),
  `num_speculative_tokens_per_batch_size`, other speculators (dflash, eagle, ngram, suffix...).
