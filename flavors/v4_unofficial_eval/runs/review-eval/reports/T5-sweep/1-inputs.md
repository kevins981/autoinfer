# T5 sweep, step 1: the framework's inputs for this task

Sources: config.json (HF hub copy), task/TASK.md, task/SPEED_OF_LIGHT.md, task_definition.md,
assumption_ledger.md, bank/T3-007 (champion code + eval/1).

## 1. Model (Qwen3.8-27B-FP8, `qwen3_5_text`)
| Item | Value | Per TP rank (TP=2) |
|---|---|---|
| Layers | 64 = 48 GDN (linear attention) + 16 full attention (every 4th) | all 64 on each rank |
| hidden d | 5120 | replicated residual |
| MLP (SiLU-gated) | intermediate 17408 | gate_up N=17408 (2x8704), down K=8704 |
| Full attention | 24 Q heads, 4 KV heads, head_dim 256, output gate (swish), partial RoPE 0.25 | 12 Q / 2 KV heads; qkv(+gate) N=7168; o_proj K=3072 |
| GDN | 16 key heads x128, 48 value heads x128, conv kernel 4, state fp32 (`mamba_ssm_dtype`) | 8 k-heads, 24 v-heads; in_proj_qkvz N=8192; in_proj_ba N=48; out_proj K=3072 |
| GDN state | 48 v-heads x 128 x 128 fp32 per layer per request | 24 heads: 1.57 MB/layer/req; x48 layers = 75.5 MB/req; x16 = 1.21 GB |
| vocab | 248,320, untied lm_head bf16 | 124,160 rows: 1.27 GB/rank |
| MTP | 1 extra layer (full-attention type + MLP) shares embedding; own lm_head path | ~0.2 GB FP8 + lm_head (FP8 in champion: ~0.64 GB/rank) |
| Weights | FP8 e4m3, 128x128 block scales | body ~12.2 GB/rank (my count: 48x191.4 MB + 16x186.1 MB = 12.17 GB) |

Per-layer FP8 weight bytes per rank: GDN layer 57.7 MB (qkvz 41.9 + out 15.7) + MLP 133.7 MB
(gate_up 89.1 + down 44.6); attention layer 52.4 MB (qkv 36.7 + o 15.7) + MLP 133.7 MB.

## 2. Hardware (per GPU, 2 GPUs)
| Resource | Value | Notes |
|---|---|---|
| F (FP8 tensor, fp32 accumulate) | ~165 TFLOP/s dense spec | fp16-acc 330 not allowed across K-blocks (TASK rule); bf16 ~165 |
| BW (GDDR6X) | ~1.0 TB/s spec; achievable ~0.85-0.9 | |
| L2 | 72 MB | the champion already prefetches 16 MB of the next GEMM's weights into it |
| C | 24 GB | weights 12.2 + lm_head 1.27 + drafter ~0.85 + KV 4.4 GiB + GDN state pool + graphs/activations |
| Link | no P2P; via host over PCIe 4.0 x16, ~25 GB/s/direction; per-message latency dominates small all-reduces | champion: ~41 us per 42-row AR (champion.md, job 25) |
| N_ridge (FP8 weights, fp32 acc) | 165e12 / (2 x 1e12) ~ 80 rows; with achievable BW ~ 95 | decode M=48 is below: memory-bound; prefill chunks (hundreds-2048 rows) compute-bound |

## 3. Workload and objective
- Official benchmark: 400 ShareGPT prompts, closed loop at c=16, temperature 0; output tok/s, mean of
  2 fresh starts. Champion trial 1 (bank/T3-007/eval/1/trial1/bench_c16.json): 104,526 input tokens
  (mean 261), 91,079 output tokens (mean 228), duration 106.6 s, 38,883 drafts (= request-steps),
  acceptance length 2.34 (pos1 0.78, pos2 0.56), median ITL 31.1 ms, mean ITL 40.6 ms.
- Derived snapshot: ~16 requests in flight, each with ~260 known prompt tokens + growing output;
  every decode step verifies 3 rows/request (M=48). Per request 1 prefill of ~260 tokens arrives
  each time another request finishes (~3.75 req/s), so roughly 1 of every ~8 steps carries prefill.
- Gates: TTFT p99 <= 1609 ms, TPOT p99 <= 55.3 ms (BASE x1.2); GSM8K-250 >= 89.6%; 128K request fits.

## 4. Contract (what counts as the same computation)
Same function up to fp reordering. Fixed: FP8 weights, FP8 KV, bf16 activations or the checkpoint's
dyn per-token-group-128 W8A8 (fp32 cross-block accumulation), baseline-precision communication (no
quantized all-reduce), target lm_head as baseline (bf16), attention/GDN math unchanged, exact spec
decode (draft-side changes free as long as verification is exact; FP8 draft lm_head allowed;
reduced-vocab draft head allowed with non-benchmark frequencies; retrained draft head allowed with
non-benchmark data). Prefix caching on; parsers on; max-num-seqs >= 16; nothing fitted to the benchmark.

## 5. Starting point: the champion's design (T3-007, 853.6 tok/s), as read from its code
| Graph part | Champion choice |
|---|---|
| Split/Place | TP=2 symmetric (heads, MLP intermediate, vocab); all 64 layers + lm_head on both ranks |
| GEMMs | W8A8: activation dyn-128 FP8 quantized inside the preceding fused norm (gemma_add_rms_norm_dyn128) or SiLU-mul (silu_and_mul_per_block_quant); Triton block-scaled FP8 GEMM with tuned tile JSONs for N=8192/7168/17408 (K=5120), N=5120 (K=3072/8704) |
| All-reduce | 2 per layer (after attention/GDN out-proj, after down-proj) = 128/step + lm_head/drafter; HMAR (host-mapped pinned buffer, fp32 sum, bitwise = NCCL) for <=96 rows, PyNCCL above |
| Overlap | side-stream L2 prefetch of the next GEMM's first 16 MB during each decode-size HMAR |
| Attention | TRITON_ATTN, bf16 Q, FP8 KV dequantized in-kernel; GDN kernels: vLLM defaults (fla-style chunk for prefill, fused recurrent for decode) |
| Speculation | MTP, 2 draft tokens (chain), FP8 draft lm_head, draft-side local argmax; batch-sharded sampling (vocab-sharded verify, no full-logits all-gather) |
| Window/batching | continuous batching, max 16 seqs, chunked prefill (default token budget), CUDA graphs: FULL for decode, piecewise for mixed steps up to 2048 tokens |
| Keep | FP8 KV cache 4.4 GiB (235,650 tokens), GDN fp32 state per request, prefix caching on |
