# T1 sweep, step 1: the framework's inputs for this task

Sources: `config.json` of Qwen/Qwen3.8-27B-FP8 (HF hub), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, BASE official results and server log
(task/baseline/official/trial1/server.log), BASE nsys profile (task/baseline/nsys_c16).

## 1. Model (per scalar index; config.json `text_config`)
| | value |
|---|---|
| layers `l` | 64 = 48 GDN (gated delta-net linear attention) + 16 full attention (every 4th: l = 3,7,..,63) |
| hidden `d` | 5120; RMSNorm (eps 1e-6), residual stream bf16 |
| MLP `m` | 17408, SiLU-gated (gate_up 5120 -> 2x17408, down 17408 -> 5120) |
| full attention | 24 q heads `h`, 4 KV heads `g` (GQA 6:1), head_dim 256, partial RoPE (25% of dims, mrope sections), q/k RMSNorm, sigmoid output gate (`attn_output_gate`) |
| GDN | 16 key heads (dim 128), 48 value heads (dim 128), conv1d kernel 4, state `S[r,l,hv,128,128]` fp32 (`mamba_ssm_dtype: float32`), gates a,b per value head |
| vocab `V` | 248,320; untied embedding and lm_head (bf16) |
| MTP draft | 1 extra layer (`mtp_num_hidden_layers: 1`, full-attention + MLP), shares the target's embedding/lm_head |
| weights | FP8 e4m3, 128x128 block scales; checkpoint 28.75 GiB (+6.04 GiB for the MTP/other shard); loaded 14.4 GiB per GPU at TP=2 |

Per-layer FP8 weight bytes per GPU at TP=2 (from shapes; checked against ncu dram bytes in
task/baseline/ncu_marlin/marlin_c16_key.csv):
- GDN in_proj (qkvz: 2048+2048+6144+6144 = 16384 out) / 2 = 8192 x 5120 = 41.9 MB (ncu: 43.2 MB read)
- GDN in_proj_ba (96 out, bf16, unquantized) tiny; GDN out_proj 3072 x 5120 = 15.7 MB (ncu 16.3 MB)
- full-attn qkv (24x256 q + gate 24x256 + 2x4x256 kv)/2 ... ~ 6656 x 5120 = 34 MB; o_proj 3072 x 5120 = 15.7 MB
- MLP gate_up 17408 x 5120 = 89.1 MB (ncu 91.2 MB); down 8704 x 5120 = 44.6 MB (ncu 46.2 MB)
- => ~190 MB per GDN layer, ~184 MB per attention layer; x64 = **~12.2 GB per GPU per forward** (agrees with SPEED_OF_LIGHT)
- lm_head (vocab-parallel) 124,160 x 5120 bf16 = **1.27 GB per GPU**; embedding same size (gathered rows only)

## 2. Hardware
2x RTX 4090 (sm_89), 24 GB each; DRAM ~1.0 TB/s spec (measured Marlin streams at 0.65-0.8 TB/s);
FP8 tensor (fp32 accum) 165-330 TFLOP/s (spec ambiguity noted in SPEED_OF_LIGHT; Ada whitepaper says
330 dense fp8/fp32-acc for 4090, bf16/fp32-acc 165), bf16 ~165 TFLOP/s;
**no GPU peer-to-peer**: inter-GPU traffic goes through host memory over PCIe 4.0 x16 (~25 GB/s per
direction each GPU). vLLM's custom all-reduce is disabled for this reason (server log: "platform lacks
GPU P2P capability"), all-reduce = NCCL 2.30.7 ring, LL protocol via SHM.
Host: 48 cores, 125 GB RAM (a third memory tier: pinned host memory, ~25 GB/s per GPU).
Framework hardware vector per GPU: F ~165 (bf16) / ~330 (fp8) TFLOP/s, BW ~1 TB/s (0.7-0.8 achieved),
Link_BW ~25 GB/s/direction **via host**, C = 24 GB.
Ridge point: bf16 weight-only Marlin (b_W = 1 B, compute bf16 165 TF): N_ridge = 165/(2*1) ~ 80 rows;
FP8 W8A8 (330 TF): ~165 rows.

## 3. Workload and objective
- Official: ShareGPT 400 prompts, input mean ~260 tokens (with chat template), output = reference
  length (mean ~228), closed loop c=16, temperature 0, `/v1/chat/completions`. Score = output tok/s
  (mean of 2 fresh-server trials). Gates: TTFT p99 / TPOT p99 <= 1.2x this run's BASE; GSM8K-250 >= 89.6%;
  one 128K-token request must work.
- Snapshot (steady state): 16 requests, each with ~260 + (0..~230) tokens known, 13-16 in decode at
  any step; about one request in 8 decode steps is replaced by a new prompt (prefill of ~260 tokens
  mixed with the others' decode).
- Derived from the BASE profile (64 prompts): 501 decode steps, 50 prefill-containing steps;
  scaled to 400 prompts: ~3,100 decode steps and ~310 mixed steps.

## 4. Contract (what counts as the same computation)
From TASK.md + task_definition.md + assumption_ledger.md:
- Same function up to fp reordering. FP8 weights, FP8 KV cache, bf16 activations, OR the checkpoint's
  W8A8 scheme (dynamic per-token-group-128 e4m3 activations vs the 128x128 block weights, fp32
  accumulation across K-blocks), per layer or per batch size; then a same-job GSM8K-250 A/B within
  2.0 points of a bf16 reference of the same code.
- Communication at baseline precision (bf16 all-reduce; no quantized comms).
- Residual stream, norms, attention/GDN math, KV, target lm_head unchanged by the W8A8 ruling.
- Speculative decoding: method and draft count free if exact (greedy-identical). Draft side:
  FP8 draft lm_head allowed; reduced-vocab draft lm_head allowed if the vocab subset comes from
  non-benchmark text; retrained draft head allowed on non-benchmark data.
- Parallelism (TP/PP/hybrid), kernels, scheduler, flags, env defaults in code: open. max-num-seqs >= 16.
- Nothing fitted to the benchmark (ShareGPT, its prompts, seed 0).
- In framework terms: Rewrite (p=1) is always allowed; Speculate only in **exact** mode (Verify runs
  the target); Keep/Recompute must not drop precision of kept state (fp32 GDN state, fp8 KV);
  Split/Place free; Window free.

## 5. Starting point: the champion's design (= BASE, read from code, server log and profile)
- Place/Split: TP=2 on every layer (Megatron column/row split: heads `h,g` and `hv` split, MLP `m` split,
  2 all-reduces per layer = 128 per forward + those of the drafter), vocab-parallel lm_head with an
  **all-gather of full logits** (42 rows x 124,160 x bf16 = 10.4 MB per GPU) before sampling.
- Rewrite / kernels: FP8 weights through Marlin (weight-only FP8, bf16 tensor-core compute) for all
  big linears; in_proj_ba and lm_heads in bf16 cutlass/cublas; FlashInfer attention (paged, fp8 KV,
  block size 1600 tokens to match the mamba page); GDN decode via `gdn_decode_post_conv_mtp_kernel`
  (fp32 state, handles the 3 verify rows), prefill via chunked FLA kernels; fused add+RMSNorm.
- Keep: FP8 KV for 16 attention layers; fp32 GDN state + conv state per request for 48 layers;
  prefix caching on (mamba "align" mode), 5 GiB KV budget (267,721 tokens), max context 131072.
- Speculate: MTP, 2 draft tokens, chain. Acceptance length 2.345 (per-position 0.78 / 0.56).
  Each draft step runs the MTP layer + the **full bf16 target lm_head** (vocab-parallel + all-gather).
  Server log: FlashInfer "cannot update draft decode metadata in place; falling back to rebuilding
  attention metadata eagerly".
- Window: one engine step = [target forward on 16x3 rows] -> [lm_head, gather, rejection sampling]
  -> [2 draft steps], all in breakable CUDA graphs; mixed steps = chunked prefill (max 2048 scheduled
  tokens) of the arriving prompt + the decoding requests in one forward.
- Measured anchor: decode step ~38.5 ms (nsys) ~ client median ITL 39.4 ms; mixed step ~147 ms
  (nsys); official 635.9 tok/s.
