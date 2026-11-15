# T3 sweep, step 1: the framework's inputs for this task

Sources: model `config.json` (HF hub, text_config), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, champion T1-014 (`bank/T1-014/change.diff`,
`serve_args.txt`, `official/1/`).

## Input 1: the model (Qwen3.8-27B-FP8, `qwen3_5_text`)
| Item | Value |
|---|---|
| Layers `l` | 64 = 48 GDN (gated delta-net linear attention) + 16 full attention (every 4th: l = 3,7,...,63) |
| Hidden `d` | 5120 |
| MLP `m` | 17408 (SwiGLU: gate_up 5120 -> 2x17408, down 17408 -> 5120) |
| Full attention | 24 query heads x 256 (+ a 24x256 sigmoid output gate in the same projection), 4 KV heads x 256 (GQA 6:1), partial RoPE (25% of 256 = 64 dims, mrope), qkv_proj 5120 -> 14336, o_proj 6144 -> 5120 |
| GDN | 16 key heads x 128 (q,k), 48 value heads x 128 (v, z gate), conv1d kernel 4 over q,k,v (10240 ch); in_proj_qkvz 5120 -> 16384, in_proj_ba 5120 -> 96; out_proj 6144 -> 5120; state `S[r,l,h]` 128x128 per value head, **fp32** (`mamba_ssm_dtype`) |
| Vocab `V` | 248,320; embedding and lm_head untied (bf16, 2.54 GB each) |
| MTP draft | 1 extra layer (attn + MLP, FP8) + fc on [embed; hidden] + lm_head; run 2 draft steps per decode step |
| Weights | FP8 e4m3, 128x128 block scales. Per layer ~384 MB (GDN) / ~372 MB (attn) -> ~24.4 GB body, **12.2 GB per GPU at TP=2** + lm_head 1.27 GB per GPU (bf16) |

Per-token, per-GPU (TP=2) sizes that matter later:
- GDN state: 24 value heads x 128 x 128 x 4 B = **1.57 MB per layer per request** -> x48 layers x16 requests = **1.2 GB per GPU**.
- KV cache (fp8): 2 KV heads x 256 x 2 (k,v) x 1 B = 1 KB per token per attn layer -> 16 KB per token per GPU. At ~16 x 400 tokens of live context: ~0.1 GB per GPU.
- Activation row: 5120 bf16 = 10 KB (one all-reduce payload per row).

## Input 2: the hardware (2x RTX 4090, sm_89)
| Resource | Value | Note |
|---|---|---|
| HBM (GDDR6X) BW | ~1.0 TB/s per GPU (spec 1008 GB/s) | measured effective probably ~0.85-0.9 (calibrate) |
| Tensor compute | bf16 fp32-acc 165 TFLOP/s; FP8 fp32-acc 330, fp16-acc 660 (Ada whitepaper, dense) | SPEED_OF_LIGHT lists FP8 fp32-acc as 165 ("check"); the whitepaper says 330. Both are far above what M=48 decode needs, so this matters only for prefill. |
| Capacity | 24 GB per GPU; gpu-mem-util 0.92 | weights 12.2 + lm_head 1.27 + embed 1.27 + MTP ~0.8 + KV 5 GiB + GDN state + graphs |
| Link | no P2P. PCIe 4.0 x16 through host memory: ~25 GB/s per direction; latency of a mapped-host-memory round trip ~ a few us | the champion's all-reduce goes through a POSIX shm buffer mapped into both GPUs (`hostmem_ar.cu`) |
| Ridge (framework's N) | FP8 weights b_W=1: N_ridge = F/(2 BW) = 330e12/2e12 = **~165 rows** (fp32-acc FP8) ; bf16 lm_head b_W=2: N_ridge = 165e12*2/(2*1e12) = ~165 rows | So a decode step (M=48 rows) is ~3.4x below the ridge: weight-bandwidth-bound. Prefill chunks (M >= 256) are compute-bound. |
| CPU | 48 cores, 125 GB RAM | engine scheduler + 2 TP worker processes |

## Input 3: the workload and objective
- Official bench: 400 ShareGPT prompts, closed loop, 16 concurrent, /v1/chat/completions, T=0, seed 0.
  Measured on the champion (`bank/T1-014/official/1/trial1`): 104.5K input tokens (mean 261),
  90.9K output tokens (mean 227), duration **108.0 s**, 841.6 tok/s.
- Snapshot at steady state: ~16 requests in decode, each with `k,v[1..n_r]` (attn layers) and
  `S[r,l]` (GDN layers) kept; context n_r ~ 260-700; every few steps a new request arrives with
  ~260 ready prompt positions (prefill chunk mixed into a decode step).
- Spec decoding: MTP, 2 drafts. Champion's acceptance: 52.2K accepted of 78.0K drafted (trial 1
  metrics): position-1 acceptance p1 = 30333/39010 = **0.778**, position-2 conditional
  p2 = 21851/30333 = **0.720**; tokens per request-step = 1 + 0.778 + 0.560 = **2.34**.
- Engine steps (trial-1 `iteration_tokens` histogram incl. warmup): 2726 steps, i.e. **~39.6 ms
  per step on average** over the run; 1807 steps of 33-64 tokens (full decode, 16x3=48 rows),
  ~540 steps of <=32 tokens (tail, partly-filled batch), ~234 steps > 128 tokens (contain a prefill
  chunk). Median ITL 33.4 ms (= one decode step as seen by a request), median TPOT 17.6 ms.
- Objective: max output tok/s. Gates: TTFT p99 / TPOT p99 within +20% of BASE (1341 / 46.1 ms;
  champion 883 / 31.6 ms: lots of room), GSM8K-250 >= 89.6% (champion 93.6%), 128K request.

## Input 4: the contract (what is "the same computation")
- Same function up to floating-point reordering; FP8 weights; FP8 KV; activations bf16 or the
  checkpoint's dynamic per-token-group-128 FP8 (W8A8, fp32 accumulation across K-blocks).
- Communication at baseline precision (bf16 all-reduce, no quantized comm).
- Exact speculative decoding only (greedy verify against the target); the draft side may change
  (fp8 draft lm_head allowed, reduced-vocab draft head allowed with non-benchmark data, retrained
  draft allowed with non-benchmark data).
- Norms, attention/GDN math, GDN state precision (fp32), target lm_head (bf16) unchanged.
- Nothing fitted to the benchmark. max-num-seqs >= 16. 128K request must still work.

## Starting point: the champion's design (T1-014), as read from its code
| Knob site | Champion's choice |
|---|---|
| Split/Place weights | TP=2 symmetric (column/row parallel), lm_head vocab-split across GPUs |
| Rewrite: GEMM precision | **all** FP8-block linears run W8A8 through the Triton block-scaled kernel (`_prefer_triton_fp8_block` default on for sm_89), at every batch size, with tuned Triton configs for the 5 shapes (N=8192/7168/17408 K=5120, N=5120 K=3072/8704) |
| Combine nodes (all-reduce) | 2 per layer (after out_proj/o_proj, after down_proj), bf16. Small messages (<=1.31 MB) via a custom mapped-host-memory kernel (`hostmem_ar.cu`: each rank writes its slot of a shm buffer, flags, reads the peer's, sums in fixed order); fused with the following residual-add + RMSNorm + FP8 quant (`fused_allreduce_gemma_rms_norm_quant`, already in the engine) |
| Window: AR overlap | Only in steps >= 256 tokens: **row-split** the batch into two halves; each half's all-reduce runs on a side stream while the other half's GEMMs run (`rowsplit.py`). Decode steps (48 rows): no overlap; ARs are on the critical path. |
| Speculate | MTP, 2 drafts, chain; fp8 draft lm_head; draft argmax by local shard + reduction; target greedy verify by local argmax per vocab shard + all-gather of (value,id) pairs (no full-logit gather) |
| Keep/Recompute | KV cache fp8 for attn layers; GDN keeps fp32 state S (in-place overwrite, with MTP-verify handling in `gdn_decode_post_conv_mtp_kernel`) |
| Attention backend | TRITON_ATTN |
| Window: batching | continuous batching + chunked prefill, max 16 seqs; CUDA graphs captured up to 512 tokens (sizes list in serve_args) |
| Sampling | batch-sharded sampling |
