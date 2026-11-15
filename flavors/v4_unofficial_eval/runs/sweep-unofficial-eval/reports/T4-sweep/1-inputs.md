# T4 sweep, step 1: the framework's inputs for this task

Sources: Qwen/Qwen3.8-27B-FP8 config.json (hub), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, HEAD = T3-004 (change.diff + serve_args.txt, workspace
`t4-head`), CONFIRMED T1-019 official results (bank/T1-019/official/1).

## 1. Model (config.json, text part)
| item | value |
|---|---|
| layers | 64 = 48 GDN (Gated DeltaNet, linear attention) + 16 full attention (every 4th: l = 3,7,...,63) |
| hidden d | 5120, RMSNorm eps 1e-6 |
| MLP | SiLU-gated, intermediate 17408 (dense, no MoE) |
| full attention | 24 Q heads, 4 KV heads (GQA 6:1), head_dim 256, output gate (q proj doubled), partial RoPE 25% (64 dims) |
| GDN | 16 K heads x 128, 48 V heads x 128 (3 V heads per K head), conv1d kernel 4, state 48 x 128 x 128 per layer per request, fp32 (`mamba_ssm_dtype`) |
| vocab | 248,320, untied lm_head (bf16) |
| MTP | 1 extra layer (full-attention layer + MLP), shares embeddings and (by default) lm_head |
| weights | FP8 e4m3, 128x128 block scales; activations dynamic per-token-group-128 (checkpoint scheme) |

Per GPU at TP=2 (FP8 = 1 B/param):
| tensor (per layer, per GPU) | shape [N x K] | MB |
|---|---|---|
| GDN in_proj_qkvz | 8192 x 5120 | 41.9 |
| GDN in_proj_ba | 48 x 5120 | 0.25 |
| GDN out_proj | 5120 x 3072 | 15.7 |
| attn qkv_proj (q+gate,k,v) | 7168 x 5120 | 36.7 |
| attn o_proj | 5120 x 3072 | 15.7 |
| MLP gate_up | 17408 x 5120 | 89.1 |
| MLP down | 5120 x 8704 | 44.6 |
| GDN layer total | | 191.5 |
| attention layer total | | 186.1 |
| **body, 64 layers** | | **12,170 MB** |
| target lm_head (bf16, vocab-sharded) | 124,160 x 5120 | 1,271 |
| MTP layer (attn + MLP, fp8) | | ~186 + fc 2*5120->5120 |
| HEAD draft head: fp8, ids < 98,304 | 49,152 x 5120 | 252 per draft step |
| GDN state per request per layer (fp32) | 24 heads x 128 x 128 x 4 B | 1.57 |
| GDN state, 48 layers x 16 requests | | 1,208 |
| conv state per request per layer | (2048+2048+... ) x 3 rows, bf16 | ~0.06 |
| KV per token, full attn (fp8, per GPU) | 16 layers x 2 KV heads x 256 x 2 (k,v) | 16 KB/token |

FLOPs per token per GPU (body): 2 x 12.17 G = ~24.3 GFLOP; + lm_head 2 x 0.636 G = 1.27 GFLOP.

## 2. Hardware
2x RTX 4090 (sm_89, 128 SMs, 24 GB each). DRAM ~1.0 TB/s spec (achievable ~0.85-0.92 for
streaming GEMMs). FP8 tensor 165 TFLOP/s (fp32 accumulate), bf16 tensor ~165 TFLOP/s dense with
fp32 accumulate (330 with fp16 acc). No P2P: GPU-GPU via host memory over PCIe 4.0 x16
(~25 GB/s per direction), and per-transfer latency of several us. L2 72 MB per GPU (relevant:
a 48-row activation tile and small tensors fit; a 40-90 MB weight matrix does not).
Ridge point for W8A16 (bf16 math, fp8 weights): N_ridge = 1 B x 165T / (2 x 1T) ~ 80 rows.

## 3. Workload and objective
ShareGPT 400 prompts, input ~260 tokens (chat template), output = reference reply length (mean
~228), temperature 0, closed loop c=16. Metric: output tok/s (mean of 2 fresh-server trials).
Snapshot of a steady decode step: 16 requests, each with KV/state kept for ~260-700 positions,
1 ready token + 2 MTP drafts = M = 48 rows. Mixed steps: a new request's ~260-token prompt
(chunked prefill, budget 8192 by default) arrives roughly every ~0.25 s
(400 requests / ~100 s), i.e. about one prefill every ~9 decode steps.

## 4. Contract (what is the same computation)
Same function as BASE up to fp reordering; FP8 weights + FP8 KV; bf16 activations or the
checkpoint's own W8A8 (per-token-group-128 dynamic, fp32 cross-block accumulation);
communication at baseline precision (bf16); exact spec decode only (greedy verify against the
target); prefix caching stays on; no tuning on benchmark data; parsers on; max-num-seqs >= 16;
128K request must fit. Gates: GSM8K-250 >= 89.6%, p99 TTFT/TPOT within +20% of BASE.

## 5. Starting point: HEAD T3-004 as read from its code
Serve: TP=2, max-len 131072, max 16 seqs, FP8 KV 5 GiB, MTP k=2 with
`draft_lm_head_quantization=fp8`, `use_local_argmax_reduction`, `--enable-batch-sharded-sampling`,
prefill CUDA graphs captured up to 512 tokens.
Design points already chosen (in framework terms):
- **Rewrite / kernels, linear layers:** Marlin repack replaced by a Triton W8A16 GEMM on the
  checkpoint layout with a fixed per-(N,K,M-bucket) tile table (decode M = 48); for M >= 80
  (prefill / mixed) W8A8 block-fp8 (checkpoint activation scheme). Marlin only for leftovers
  (and a tuned mid-M thread config).
- **Place/communication:** all-reduce of <= 128 rows goes through a custom pinned-host-memory
  2-rank all-reduce (CUDA-graph capturable), larger ones through NCCL; prefill GEMM+AR row
  chunking (chunk k's AR overlaps chunk k+1's GEMM, >= 192 rows, eager only).
- **Keep/Recompute on the GDN state (ReplaySSM, framework 02 §4.3 row B):** "replay mode":
  during an MTP verify step the state is not written per token; the accepted rows' (k, v, decay,
  beta) are kept as pending rows and replayed at the next step (single state write), with a
  "lazy" variant that commits only every other step (2 pending pages). Fused GDN decode kernel
  (conv + gating + delta rule) in CUDA.
- **Speculate:** MTP 2 drafts (acceptance ~2.32-2.34 tokens/step); draft lm_head FP8 over the
  first 98,304 token ids (reduced, generic-vocab subset) ; greedy verify from per-row target argmax
  computed vocab-parallel (local argmax + tiny all-gather) instead of gathering full logits.
- **Attention:** FlashInfer for full attention, with device-side exact lengths (no blocking
  seq_lens .cpu() sync in planning).
- CUDA graphs for decode and for prefill up to 512 tokens.
Measured: CONFIRMED T1-019 official 897.5 tok/s, median ITL 29.1 ms, median TPOT 16.3 ms,
acceptance 2.32; HEAD T3-004 unofficial +2.1% over it (expected ~916).
