# T58 sweep, step 1: the framework's inputs for this task

Written in phase 1 (no run history read: only framework/, task/, task_definition.md,
assumption_ledger.md, champion.md and bank/T52-005/).

## 1. Model (config.json of Qwen/Qwen3.8-27B-FP8, text part)

| Quantity | Value |
|---|---|
| Layers L | 64 = 48 GDN (gated delta-net, "linear_attention") + 16 full attention (every 4th: l = 3,7,...,63) |
| Hidden d | 5120 |
| MLP m | 17408 (SiLU-gated: gate_up 2x17408, down 17408 -> 5120) |
| Full attention | 24 query heads, 4 KV heads, head_dim 256, output gate (q projection doubled: 2x24x256), partial RoPE 0.25 (64 of 256 dims), mrope |
| GDN | 16 key heads x 128, 48 value heads x 128 (GVA 3:1), conv1d kernel 4 over q,k,v (2048+2048+6144 = 10240 channels), recurrent state per value head 128x128 fp32 ("mamba_ssm_dtype": float32), gated (z) RMSNorm on output |
| Vocab V | 248,320; untied embedding and lm_head (bf16) |
| MTP | 1 layer (fc 2x5120 -> 5120, 1 full-attention decoder layer, norm), shares the lm_head |
| Quantization | FP8 e4m3, 128x128 block-scaled weights; activation scheme "dynamic" (allowed: per-token-group-128 W8A8) |

### Per-rank shapes at TP=2 (N x K, FP8 bytes = N*K)

| Node | N x K per rank | MB per rank per layer |
|---|---|---|
| GDN in_proj_qkvz | 8192 x 5120 | 41.9 |
| GDN in_proj_ba (b, a: 48 heads each) | 48 x 5120 | ~0.5 (bf16) |
| GDN out_proj | 5120 x 3072 | 15.7 |
| Attn qkv (q+gate, k, v) | 7168 x 5120 | 36.7 |
| Attn o_proj | 5120 x 3072 | 15.7 |
| MLP gate_up | 17408 x 5120 | 89.1 |
| MLP down | 5120 x 8704 | 44.6 |
| **GDN layer total** | | **191.8** |
| **Attention layer total** | | **186.1** |
| **Body, 64 layers** | | **12.18 GB** |
| Target lm_head (bf16, V/2 x 5120) | 124160 x 5120 | 1271 (bf16) |
| MTP layer (fc + attn layer) | | ~212 (fp8) |
| Draft lm_head (champion: first 98304 ids, fp8, V'/2 per rank) | 49152 x 5120 | 252 |

FLOPs per row (token) per rank: 2 x 12.18e9 = 24.4 GFLOP body (+1.27 GFLOP target lm_head).

### Per-request kept state (per rank)
| Data node | Size per request per rank |
|---|---|
| GDN recurrent state S[l] (24 value heads x 128x128 fp32) | 1.57 MB per layer; x48 = 75.5 MB |
| GDN conv state (3 x 5120 bf16) | 30 KB per layer; x48 = 1.5 MB |
| KV cache (fp8), 2 KV heads x 256 x (k,v) | 1 KB per token per layer; x16 layers = 16 KB per token |
| At c=16, ~500 tokens of context: S = 1.21 GB, KV = 0.13 GB per rank | |

## 2. Hardware (2x RTX 4090, per GPU)

| Resource | Value | Note |
|---|---|---|
| DRAM BW | 1008 GB/s spec (~0.9 achievable) | |
| L2 | 72 MB | big enough to hold a GEMM's weight slice (champion uses it: L2 prefetch during ARs) |
| FP8 tensor (fp32 accumulate) | ~165 TFLOP/s dense (spec, boost) | the W8A8 path; GPU 0 is locked at 2235 MHz (~-11..15% on compute) |
| BF16 tensor (fp32 accumulate) | ~82.6 TFLOP/s dense | the Marlin weight-only path (bf16 MMA) |
| Ridge N (rows per weight load), fp8 weights b_W=1 | W8A8: 165e12/(2 x 1.0e12) = **~82 rows**; Marlin bf16 MMA: **~41 rows** | framework 01 §5 with this machine's numbers (not 330) |
| Link | no P2P; via host memory over PCIe 4.0 x16: ~25 GB/s per direction (spec ~31) | every all-reduce = D2H write + H2D read through pinned host buffers |
| Copy engines | DMA copy engines usable for D2H/H2D without SMs (champion uses them for mixed steps) | |
| Host | 48 cores, 125 GB RAM; host time matters for eager (non-graph) steps | |
| Capacity C | 24 GB per GPU, 0.92 utilization; weights ~13.7 GB, KV budget 5 GiB, GDN state, graphs | |

## 3. Workload and objective (TASK.md)

- Closed loop, 16 concurrent requests, 400 ShareGPT prompts; input mean ~260 tokens (chat
  template incl.), output mean ~228 tokens; temperature 0 (greedy). Score = output tok/s (~91K
  output tokens; champion ~75 s per trial).
- Snapshot (steady state): ~16 requests, each with kept {S[l] for 48 GDN layers, conv state,
  k,v[1..n] for 16 attention layers}, 1 known-but-unprocessed token + 3 drafted tokens per
  request; occasionally a new request with ~260 known tokens and nothing kept (a "prefill").
- Goal: max throughput, latency guard (TTFT p99, TPOT p99 within +20% of BASE: 1341 / 46.1 ms
  BASE-ish; champion 667 / 26.6 ms, so there is headroom).

## 4. Contract (what counts as the same computation)

From TASK.md and task_definition.md:
- Same function up to fp reordering. Weights FP8 (block 128x128), KV FP8. Activations bf16 or
  the checkpoint's W8A8 (dynamic per-token-group-128 e4m3, fp32 accumulation across K-blocks).
- Comm at baseline precision (bf16 partials, fp32 add): no lossy compression. (Champion uses a
  bitwise-lossless re-encoding of the AR payload: allowed because it is lossless.)
- Attention/GDN math, norms, residual, target lm_head unchanged in precision (target lm_head may
  be screened by a certified bound, then exact).
- Spec decode must be exact (greedy output = target's). Drafter is free (fp8 head, reduced
  vocab, screened head, retrained head with non-benchmark data).
- Nothing benchmark-specific. Memory changes need the 128K request test.
- **Framework mapping:** Rewrite (exact over R) is allowed; Speculate only in exact mode (the
  Approximate-Verify column of 04 is closed); Keep/Recompute free; Split/Merge/Place/Window free
  within these rules (parallelism may change).

## 5. Starting point: the champion's design (T52-005), as read from its code

Serve: TP=2, MTP k=3 drafts (acceptance per position 0.757/0.53/0.376, mean accepted length
2.66 incl. bonus), TRITON_ATTN, FP8 KV, CUDA graphs for decode and for prefill/mixed sizes up to
1536 tokens.

| Graph site | Champion's choice (knob reading) |
|---|---|
| Body linears, M <= 96 rows (decode/verify: 16 x 4 = 64 rows) | W8A8: fused producer (norm->quant, silu_mul->quant) + register-direct e4m3 GEMM `t31_gemm_e4m3` with L2-regime-dependent configs (Rewrite of the precision path; Window: weight read once per step) |
| Body linears, M > 96 (prefill/mixed) | W8A8 `bs_gemm_marlin` on the Marlin layout (gate_up also on bs_gemm for 48 < M <= 96) |
| All-reduces (2 per layer + final), decode | host-staged AR through pinned /dev/shm buffers, SM kernel, bitwise-lossless block-exponent compression (~0.77 bytes) on the PCIe hop; extra blocks L2-prefetch the NEXT GEMM's weights while the AR waits on the link (Window: Link || HBM overlap) |
| All-reduces, mixed steps >= 256 rows | token split into 2 chunks (Split r/i + Window): chunk 0 via copy-engine DMA AR, chunk 1 on an aux stream; "ping-pong" carries each chunk on into the next layer's norm+projection; link priority (chunk 1's ARs wait for chunk 0's) |
| Mixed steps >= 768 rows | two layer-pipelined "waves" through all 64 layers (no per-layer join), GDN state handed over at a 64-aligned cut |
| GDN decode/verify | fused kernel; "checkpoint + replay" (Keep {S[t-L], k,v,decay,beta log} = ReplaySSM Row B of 02 §4.3) with lazy checkpoint (log cap 8 tokens): S read every step, written only when the log would overflow |
| GDN prefill core | chunked (64) GDN, CUDA-graph captured per (N<=3 seqs, <=16 chunks) |
| Target lm_head | certified screen: 7-bit integer plane in HBM (0.57 GB/rank), bound-based candidate set, exact bf16 logits of candidates from a pinned-host bf16 copy; greedy = argmax |
| Drafter | MTP layer (Marlin bf16-activation linears); draft lm_head = first 98304 ids, fp8, plus PCA screen (top-64 candidates, exact rescoring); local argmax reduction |
| Scheduling | vLLM v1 continuous batching + chunked prefill; max 16 seqs |
| NUMA | workers self-bind to their GPU's NUMA node |
