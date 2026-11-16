# T3 sweep, step 1: the framework's inputs for this task

Sources: model `config.json` (HF hub copy), `task/TASK.md`, `task/SPEED_OF_LIGHT.md`, `task_definition.md`,
`assumption_ledger.md`, `champion.md`, `bank/T1-007/{change.diff,serve_args.txt,eval/1}` (champion server log).

## 1. Model (Qwen3.8-27B-FP8, `Qwen3_5ForConditionalGeneration`, language model only)
| | value |
|---|---|
| layers | 64 = 48 GDN (linear attention, Gated DeltaNet) + 16 full attention (every 4th: l = 3,7,...,63) |
| hidden d | 5120; RMSNorm eps 1e-6; residual stream bf16 |
| MLP | SwiGLU, m = 17408 (gate+up fused 34816) |
| full attention | 24 Q heads, 4 KV heads (GQA 6:1), head_dim 256, **output gate** (q-proj emits q and gate: 2x24x256), partial RoPE 25% (64 dims, mrope interleaved), q/k RMSNorm |
| GDN | 16 key heads x 128, 48 value heads x 128 (GVA 3:1), causal conv1d k=4 over q,k,v, gates a,b (48 each), z output gate (6144), **state S[r,l,h,128,128] fp32** (`mamba_ssm_dtype float32`) |
| vocab | 248,320, untied embed / lm_head, bf16 |
| MTP | 1 extra layer (`mtp_num_hidden_layers 1`), shares embedding and lm_head with the target |
| weights | FP8 e4m3, 128x128 block scales; checkpoint 28.75 GiB + MTP 6.04 GiB file (MTP file holds bf16 parts) |

Per-GPU shapes at TP=2 (match the champion's tuned Triton configs N,K):
| GEMM (per layer, per GPU) | N x K | FP8 bytes |
|---|---|---|
| MLP gate_up | 17408 x 5120 | 89.1 MB |
| MLP down | 5120 x 8704 | 44.6 MB |
| GDN in_proj_qkvz | 8192 x 5120 | 41.9 MB |
| GDN in_proj_ba | 48 x 5120 | 0.25 MB (bf16 0.5 MB) |
| GDN out_proj | 5120 x 3072 | 15.7 MB |
| Attn qkv(+gate) | 7168 x 5120 | 36.7 MB |
| Attn o_proj | 5120 x 3072 | 15.7 MB |
Totals per GPU per forward: MLP 64x133.7 = 8.56 GB, GDN 48x57.7 = 2.77 GB, attn 16x52.4 = 0.84 GB:
**12.17 GB body**, + lm_head 248320/2 x 5120 bf16 = **1.27 GB**, + MTP layer ~0.2 GB FP8 (+ its bf16 fc 2x5120->5120, 26 MB/GPU).

Per-request state per GPU: GDN S = 24 heads x 128 x 128 x 4 B = 1.57 MB/layer -> 75.5 MB/request (48 layers) ->
**1.21 GB for 16 requests**; conv state small (3 x 5120 x 2 B / 2 per layer). KV cache fp8: 16 layers x 2 KV heads/GPU x
256 x 2 (k,v) x 1 B = 16 KB/token/GPU -> at ~500-token contexts x 16 req = 130 MB/GPU.

## 2. Hardware
2x RTX 4090 (sm_89, 24 GB, ~1.0 TB/s DRAM, FP8 TC ~165 TFLOP/s fp32-acc / ~330 fp16-acc; bf16 ~165 dense with fp32 acc),
72 MB L2, 128 SMs. **No P2P**: GPU-GPU via host over PCIe 4.0 x16 (~25 GB/s/dir). 48 CPU cores (host launch path matters
for eager steps). Measured by others (champion record): one TP all-reduce of [M,5120] bf16: NCCL 39-97 us, champion's
host-mapped AR (HMAR) 19 us (M=16) .. 46 us (M=48) .. 83 us (M=96): **latency-bound, not bandwidth-bound** (0.5 MB at
25 GB/s = 20 us; the rest is the flag round-trip over PCIe, ~2 x 1-2 us, plus the copy).

## 3. Workload and objective
`vllm bench serve`, ShareGPT 400 prompts (prompt ~260 tok incl. template, output ~228), closed loop c=16, temp 0.
Metric: output tok/s (mean of 2 fresh starts). Gates: TTFT p99 <= 1.2x1341 ms, TPOT p99 <= 1.2x46.1 ms, GSM8K-250 >= 89.6%,
one 128K request served. Promotion: mean higher and every start above the champion's mean; <1% gain needs a 3rd start.
Noise: per-start sigma 0.3-0.5%, job drift up to ~1.6%.

Workload snapshot (steady state): 16 running requests, each with ~260 prompt + up to ~228 generated tokens kept;
every step, each request has 1 + 2 draft = 3 ready rows (M = 48 rows) unless a new request is prefilling
(~260 rows, one every ~(228/2.34) ~ 97 steps per request slot, i.e. a new prefill every ~6 steps across 16 slots).

## 4. Contract (what counts as the same computation)
- Same function up to fp reordering; FP8 weights, FP8 KV; activations bf16 or the checkpoint's W8A8 (dynamic per-token
  group-128 FP8, fp32 accumulation across K-blocks); residual, norms, attention/GDN math (incl. fp32 GDN state), KV,
  **communication at bf16**, target lm_head unchanged. Exact spec decode only (draft side is free: draft lm_head fp8 ruled OK;
  reduced-vocab draft head from non-benchmark data OK; retrained draft head from non-benchmark data OK).
- Prefix caching stays on; parsers on; max-num-seqs >= 16; 128K request must fit; nothing tuned on benchmark data.

## 5. Starting point: the champion T1-007 (771.3 tok/s; starts 762.6 / 779.9; acceptance 2.340)
Read from `serve_args.txt`, `change.diff` and its server log:
- TP=2 over both GPUs, V2 model runner, `compilation mode NONE` (no torch.compile / inductor fusions), breakable CUDA graphs:
  FULL graphs for uniform decode batches (captured sizes up to 48 tokens etc.) and PIECEWISE graphs up to 2048 tokens
  (`max_prefill_cudagraph_capture_tokens 2048`, T2-001), so mixed prefill steps are graphed too.
- MTP, 2 draft tokens, greedy; drafter picks tokens by **vocab-parallel local argmax** (no full-logit gather, T2-003);
  drafter lm_head = the target's **bf16** lm_head (T1-001's fp8 draft head flag is NOT in the champion);
  FlashInfer "cannot update draft decode metadata in place; rebuilding attention metadata eagerly" (log).
- `--enable-batch-sharded-sampling` (target sampling split across ranks).
- All FP8 linears via **Triton block-scaled W8A8** (dynamic per-token-group-128 activation quant) with 5 tuned config JSONs
  (T2-004) instead of Marlin weight-only.
- All-reduce: custom host-mapped AR (HMAR) for <=96-row bf16 all-reduces (decode), NCCL above (prefill/mixed) (T1-007).
- Attention: FlashInfer (fp8 KV, bf16 query); GDN: FLA Triton prefill kernels, `gdn_decode_post_conv_mtp` decode kernel.
- KV budget 5 GiB (267,721 tokens), max_num_scheduled_tokens 2048 (chunked prefill), max-num-seqs 16.
- Measured: median ITL 35.0 ms (= one decode step delivering ~2.34 tokens/request), median TPOT 19.0 ms,
  median TTFT 180 ms; 400 requests in ~118 s per start.
