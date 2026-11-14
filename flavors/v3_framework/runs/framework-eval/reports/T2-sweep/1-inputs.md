# T2 sweep, step 1: the framework's inputs for this task

Sources: config.json (Qwen/Qwen3.8-27B-FP8), task/TASK.md, task/SPEED_OF_LIGHT.md, task_definition.md,
assumption_ledger.md, BASE server log + nsys profile (task/baseline/nsys_c16), vLLM code at 4527515cd7.

## 1. Model (per config.json, text model only)
| item | value |
|---|---|
| layers L | 64 = 48 GDN (linear attention, Gated DeltaNet) + 16 full attention (every 4th) |
| hidden d | 5120; MLP m = 17408 (SiLU-gated: gate_up 2x17408, down 17408) |
| full attention | 24 q heads x 256 (with output gate: q_proj outputs 2x), 4 KV heads x 256, partial RoPE (25%), q/k RMSNorm |
| GDN | 16 key heads x 128, 48 value heads x 128, conv kernel 4, state S[h_v, 128, 128] in **fp32** (mamba_ssm_dtype) |
| MTP draft | 1 extra layer (full-attention block + fc over [embed; hidden]) shares embed + lm_head |
| vocab V | 248,320, lm_head untied, bf16 (2.54 GB total) |
| weights | FP8 e4m3 128x128 block scales; all linear layers of the body + MTP layer |

Per GPU at TP=2 (FP8 bytes): GDN layer: in_proj_qkvz 8192x5120 = 42 MB, out_proj 3072x5120 = 15.7 MB;
attn layer: qkv(+gate) 7168x5120 = 36.7 MB, o_proj 15.7 MB; MLP: gate_up 17408x5120 = 89 MB,
down 8704x5120 = 44.6 MB. Body total ~12.15 GB/GPU. lm_head 1.27 GB/GPU (bf16, vocab-split).
GDN state per request per layer per GPU: 24 heads x 128x128 x 4 B = 1.57 MB (x48 layers x16 req = 1.2 GB).

## 2. Hardware (per GPU, x2)
| | value | note |
|---|---|---|
| DRAM BW | 1008 GB/s spec; Marlin achieves 0.63-0.80 TB/s (ncu, BASE) | |
| L2 | 72 MB | big enough to hold one ~40 MB GEMM weight (relevant for prefetch ideas) |
| tensor | bf16 (fp32 acc) ~165 TF dense; FP8 (fp32 acc) 2x that per Ada whitepaper (SOL file says 165: "check") | |
| ridge N | bf16 compute on FP8 weights: 165e12/1e12 = 165 flop/B -> ~82 rows; FP8 MMA: ~165 rows | |
| link | no P2P; GPU<->GPU via host memory, PCIe 4.0 x16 ~25 GB/s/direction | |
| capacity | 24 GB; 0.92 util; weights 12.2 + lm_head 1.27 + embed + MTP + 5 GiB KV + GDN state pool | |

## 3. Workload and objective
ShareGPT 400 prompts, mean ~260 input tokens (with chat template), mean ~228 output, closed loop c=16,
T=0. Score = output tok/s (BASE 635.9). Gates: TTFT p99 / TPOT p99 <= 1.2x BASE (1341 / 46.1 ms),
GSM8K-250 >= 89.6, 128K request works.
Derived (from BASE profile, 64 prompts): two step kinds.
- decode steps (~89% of steps): ~15-16 requests x 3 rows (1 verified + 2 drafts) = M~45 rows; 38.5 ms.
- prefill-containing steps (~1 per finished request): median 215 prefill + 45 decode rows (M~260); 147 ms.
- Full benchmark estimate: ~2,500 decode steps (~98 s) + ~310 prefill steps (~45 s) = ~143 s
  (91.2K output tokens / 635.9 tok/s = 143 s). **Prefill steps are ~1/3 of wall time.**

## 4. Contract (what counts as the same computation)
Same function up to fp reordering; FP8 weights + FP8 KV fixed; activations bf16 or the checkpoint's
own W8A8 (dynamic per-token-group-128 FP8 acts, fp32 cross-block accumulation; requires a same-job
GSM8K-250 A/B within 2 points); communication at bf16; norms/attention/GDN/KV/target lm_head unchanged;
exact speculative decoding only (draft-side changes like FP8 draft lm_head or a reduced-vocab draft
lm_head from non-benchmark frequencies are allowed); parsers on; max-num-seqs >= 16; nothing fitted to
the benchmark. GDN state dtype is fp32 in the model config: I treat it as part of the "GDN math"
(not changeable).

## 5. Starting point: the champion's design (BASE = the task baseline, no diff)
- TP=2 everywhere (Megatron split: column-parallel in_proj/qkv/gate_up, row-parallel out/o/down ->
  2 bf16 all-reduces per layer = 128 per forward + embedding all-reduce (vocab-parallel embedding) +
  logits all-gather (vocab-parallel lm_head)).
- All-reduce backend: **PYNCCL only** (custom all-reduce disabled: no P2P; SymmMem unsupported on sm89;
  FlashInfer AR unsupported at world 2; **FLASHINFER_PCIE_IPC exists but is opt-in via
  VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 default**). NCCL 2.30.7.
- FP8 linear layers through Marlin (weight-only FP8, bf16 activations, bf16 MMA, fp32 accumulate).
- Attention: FlashInfer (bf16 q, fp8 KV). GDN decode: fused CUDA kernel `gdn_decode_post_conv_mtp`
  which reads the source state and **writes the state after every one of the step's tokens** (3 slots
  per request) so the next step can resume from whichever draft was last accepted. Prefill GDN: FLA
  chunked Triton kernels.
- Spec decode: MTP, 2 draft tokens, greedy drafts; draft runs 2 sequential MTP steps (first over all
  ~45 rows, second over 16 rows), each with a **full bf16 lm_head** (1.27 GB/GPU) + logits all-gather;
  verification via rejection sampler over full gathered target logits.
- CUDA graphs ("breakable" CUDA graphs auto-enabled), torch.compile fusions norm_quant/act_quant;
  max_num_scheduled_tokens 2048; chunked prefill; KV budget 5 GiB fp8; max-num-seqs 16.
