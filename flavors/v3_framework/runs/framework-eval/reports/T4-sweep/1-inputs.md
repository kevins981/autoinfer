# T4 sweep, step 1: the framework's inputs for this task

Sources: config.json of Qwen/Qwen3.8-27B-FP8 (hf hub), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, champion.md, bank/T3-009/{change.diff,serve_args.txt,official/1}.

## 1. Model (text part only; `--language-model-only`)
| Item | Value |
|---|---|
| Layers | L = 64: 48 GDN (linear attention, "linear_attention") + 16 full attention (every 4th layer: 3,7,...,63) |
| Hidden d | 5120 |
| Full attention | 24 query heads x 256, 4 KV heads x 256 (GQA 6:1), **output gate** (q_proj emits q and a gate: qkv N = 2*6144 + 2*1024 = 14336), partial RoPE (25% of dims, mrope interleaved), q/k norm |
| GDN (Gated DeltaNet) | 16 key heads x 128 (q, k), 48 value heads x 128 (v), z gate 48x128 (in_proj_qkvz N = 2048+2048+6144+6144 = 16384), in_proj_ba N = 96 (beta, decay per value head), causal conv1d width 4 on q,k,v (10240 channels), recurrent state S[h] 128x128 per value head, **fp32** (`mamba_ssm_dtype`), gated RMSNorm on output, out_proj K = 6144 |
| MLP | SwiGLU, intermediate 17408 (gate_up N = 34816, down K = 17408) |
| Norms | Gemma-style RMSNorm (1+w), eps 1e-6, before attention and before MLP |
| Vocab | 248,320, untied embedding and lm_head (bf16) |
| MTP | 1 extra layer (fc 2d->d, one full-attention decoder layer + MLP, norms), shares the lm_head in the checkpoint |
| Weights | FP8 e4m3, 128x128 block scales; activation_scheme dynamic (W8A8 per-token-group-128 allowed) |

Per-GPU weight bytes under TP=2 (FP8, 1 B/param), from the shapes:
| Matrix (per GPU) | Shape (N x K) | MB |
|---|---|---|
| GDN in_proj_qkvz | 8192 x 5120 | 41.9 |
| GDN in_proj_ba | 48 x 5120 | 0.25-0.5 |
| GDN out_proj | 5120 x 3072 | 15.7 |
| Attn qkv_proj | 7168 x 5120 | 36.7 |
| Attn o_proj | 5120 x 3072 | 15.7 |
| MLP gate_up | 17408 x 5120 | 89.1 |
| MLP down | 5120 x 8704 | 44.6 |
| GDN layer total | | ~191.5 |
| Attention layer total | | ~186.1 |
| 64 layers | 48 x 191.5 + 16 x 186.1 | **~12.17 GB** |
| Target lm_head (bf16, vocab-split) | 124,160 x 5120 x 2 B | **1.27 GB** |
| Draft (MTP) layer, FP8 | ~186 MB + fc (2d->d) | ~0.2 GB |
| Draft lm_head (champion: FP8, vocab reduced to ids [0,151643) + 33 special ids, split) | ~75.8K x 5120 x 1 B | ~0.39 GB |

Per-request kept state, per GPU:
- GDN: 24 value heads x 128 x 128 x 4 B = 1.57 MB per layer -> **75.5 MB per request** (48 layers) -> 1.21 GB for 16 requests. Plus conv state (3 x 5120 per layer, small). With MTP, each request holds extra state slots for the speculative rows.
- Full attention KV (fp8): 16 layers x 2 x 2 KV heads x 256 = 16 KB per token per GPU -> at ~500 tokens of context ~8 MB per request, 128 MB for 16. Small next to the GDN state.

## 2. Hardware (per GPU, 2 GPUs)
| Resource | Value | Note |
|---|---|---|
| DRAM BW | ~1.0 TB/s (spec 1008 GB/s); achievable ~0.85-0.92 | measured in step 3 from GEMM kernels |
| L2 | 72 MB | big enough to hold a GDN layer's in_proj (42 MB) or part of gate_up; the champion already uses L2 prefetch during the all-reduce |
| FP8 tensor | ~165 TFLOP/s fp32-accumulate, ~330 fp16-acc (dense, SoL; fp32-acc is the one the rules allow across K-blocks) | |
| BF16 tensor | ~165 TFLOP/s (fp32 acc) | (check) |
| SMs | 128 | |
| Capacity | 24 GB | 0.92 utilization |
| GPU-GPU link | **no P2P**: via pinned host memory over PCIe 4.0 x16, ~25 GB/s per direction | the champion's all-reduce writes its partial to mapped host memory and reads the peer's |
| Ridge N (FP8 weights, fp32 acc) | N_ridge = b_W F / (2 BW) = 1 x 165e12 / 2e12 ~ **80 rows** (160 with fp16 acc) | much lower than the framework's "333" example |

## 3. Workload and objective
- vllm bench serve, ShareGPT 400 prompts, input mean ~260 tokens (chat template), output = reference length, mean ~228; temperature 0; closed loop, **16 concurrent**.
- Objective: output tok/s (mean of 2 fresh-server trials). Gates: TTFT p99 and TPOT p99 within +20% of this run's BASE (BASE: 1341 / 46.1 ms), GSM8K-250 >= 89.6%, one 128K request served.
- Snapshot in steady state: 16 requests decoding with ~260-500 tokens of context; each decode step verifies 1 + 2 draft rows per request (M = 48 rows); a new request arrives whenever one finishes (400 over ~96 s: ~4 arrivals/s), so ~1 in 6-10 steps also carries a prefill chunk of ~260 tokens ("mixed" step).
- Champion official: 951.5 tok/s, MTP acceptance length 2.33 (per position 0.775, 0.556), TPOT p50 15.5 ms, median ITL 27.9 ms (= decode step period incl. mixed steps share), TTFT p99 ~805 ms, TPOT p99 ~29 ms. Total benchmark wall: 91.2K output tokens / 951.5 = **~96 s**.

## 4. Contract (what counts as the same computation)
- Same function as BASE up to fp reordering; no approximation. FP8 weights, FP8 KV cache.
- Activations: bf16, or the checkpoint's W8A8 (dynamic per-token-group-128 e4m3 activations, fp32 accumulation across K-blocks). Norms, attention/GDN math, KV, communication, target lm_head stay at baseline precision.
- Communication at baseline precision (no quantized all-reduce).
- Spec decode: exact (greedy verify against target); draft side may be changed (FP8 / reduced vocab draft head allowed; retrained draft allowed with non-benchmark data).
- Nothing fitted to the benchmark; parsers on; max-num-seqs >= 16; 128K request must fit when memory changes.

## 5. Starting point: the champion's design (T3-009), as read from its code
Parallelism and placement: **TP=2** over everything (column-parallel in_proj/qkv/gate_up, row-parallel out/o/down -> one all-reduce after the token mixer and one after the MLP: 128 all-reduces per forward), vocab-parallel lm_heads.
Knob choices visible in the diff:
| Knob | Champion's choice |
|---|---|
| Rewrite (precision path) | W8A8 block-FP8 GEMMs via Triton (`VLLM_FP8_BLOCK_PREFER_TRITON`, tuned configs for the 5 shapes) instead of Marlin weight-only; norm + group-128 quant fused (`gemma_add_rms_norm_group_quant`), SiLU-mul + quant fused |
| Place / Link | **host-memory all-reduce** (`hostmem_ar.cu`): each rank writes its bf16 partial into a cudaHostRegister'ed shm slot, flags, polls the peer flag, reads the peer slot and sums (fixed order -> bitwise identical on both ranks); for messages <= 1.25 MB; CUDA-graph capturable |
| Window (Link || HBM) | the AR kernel carries extra CTAs that **prefetch the next GEMM's weights into L2** (40 MB, evict_last) while the link exchange runs: gate_up during the attention AR, next layer's in_proj during the MLP AR |
| Split r/i + Window (mixed steps >= 256 tokens) | **row-split**: rows split in two halves, each half's AR on a side stream overlaps the other half's GEMMs (exact) |
| Keep/Recompute (GDN state) | **DeltaLog**: in a verify step, row 0 writes the full fp32 state; rows 1..2 write a small log (k, v, decay, beta: ~784 B per head) instead of a full 64 KB state; the accepted slot is materialized (replayed) on demand next step |
| Speculate | MTP, 2 draft tokens, greedy verify done locally: per-shard argmax + all-gather of (value, id) pairs (`greedy_local`, `use_local_argmax_reduction`) instead of gathering full logits |
| Draft head | FP8 draft lm_head, vocab reduced to ids < 151643 + special tokens (data-free choice from tokenizer layout) |
| Attention | TRITON_ATTN backend |
| CUDA graphs | capture sizes up to 512 tokens (incl. 48 for decode) |
| Sampling | batch-sharded sampling |

What the champion did NOT change (from the diff): TP=2 split itself, the number of ARs per layer, bf16 target lm_head (fixed), GDN state dtype fp32, chunked-prefill/scheduler policy (default), MTP draft count 2, chain (not tree) speculation.
