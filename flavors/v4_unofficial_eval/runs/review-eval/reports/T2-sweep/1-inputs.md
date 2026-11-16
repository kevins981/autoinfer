# T2 sweep, step 1: the framework's inputs for this task

Sources: model config.json (HF hub copy), task/TASK.md, task/SPEED_OF_LIGHT.md, task_definition.md,
assumption_ledger.md, champion.md (BASE), task/harness/serve_args.sh, task/baseline/ (nsys + ncu +
official result), the vLLM source at the baseline commit.

## Input 1: model (Qwen3.8-27B-FP8, `qwen3_5_text`)
| item | value |
|---|---|
| layers | 64 = 48 GDN (linear attention, Gated DeltaNet) + 16 full attention (every 4th) |
| hidden d | 5120; MLP m = 17408 (SwiGLU: gate_up 5120 -> 2x17408, down 17408 -> 5120) |
| full attention | 24 q heads, 4 KV heads, head_dim 256, output gate, partial RoPE (0.25) |
| GDN | 16 key heads x 128, 48 value heads x 128, conv kernel 4, state 48 x 128 x 128 per layer, **fp32 state** (`mamba_ssm_dtype float32`) |
| vocab | 248,320, untied lm_head (bf16, 1.27 GB total, 0.635 GB per GPU under TP=2) |
| MTP | 1 extra layer (drafter), shares the target's lm_head in vLLM; 2 draft steps per decode step |
| weights | FP8 e4m3, 128x128 block scales; checkpoint "activation_scheme: dynamic" (W8A8 per-token-group-128 is the native scheme) |
| per-GPU bytes per forward | ~12.2 GB body (FP8) + 0.635 GB lm_head |
| FLOPs per token per GPU | ~25.6 GFLOP (2 x params/2) |

## Input 2: hardware (2x RTX 4090, sm_89)
| resource | value | note |
|---|---|---|
| DRAM BW | 1008 GB/s spec; Marlin reaches 630-800 GB/s (ncu, baseline/ncu_marlin) | |
| bf16 tensor, fp32 acc | ~82.6 TFLOP/s dense | **Marlin computes in bf16**: ridge N = 82.6/1.0/2 = **~41 rows** for FP8 weights |
| FP8 tensor, fp32 acc | ~165 TFLOP/s dense | ridge ~82 rows |
| FP8 tensor, fp16 acc | ~330 TFLOP/s | allowed only inside a 128-K block per the accumulation rule |
| GPU-GPU link | **no P2P**; via host memory over PCIe 4.0 x16 (~25 GB/s/dir spec) | NCCL uses SHM transport; measured below in 3-calibration |
| capacity | 24 GB/GPU; weights 14.4 GiB/GPU loaded; KV budget fixed 5 GiB (flag) | |
| CPU | 48 cores; one Python process per TP rank launches kernels | eager steps are CPU-launch-bound |

The framework's "333-row rule" does not hold here: the ridge depends on the kernel's compute
dtype. With Marlin (FP8 weights dequantized to bf16 in registers) the ridge is ~41 rows, so the
decode step at M = 48 rows is already at the ridge and prefill/mixed steps are compute-bound.

## Input 3: workload and objective
- Official benchmark: 400 ShareGPT prompts (mean ~260 input tokens incl. template, ~228 output
  tokens), closed loop at concurrency 16, temperature 0, 2 fresh server starts; score = mean
  output tok/s. Gates: TTFT p99 / TPOT p99 not >20% worse than BASE (1341 / 46.1 ms), GSM8K-250
  >= 89.6%, one 128K request must work. Promotion: mean of all harness runs > champion and every
  start above champion mean; a <1% gain needs one more start in a new job (TRIALS=1), all 3 starts above the champion mean (ruling 2026-11-16).
- Snapshot at steady state: 16 requests, each with ~260-500 known tokens, kept state = KV of 16
  attention layers (fp8) + GDN state S (fp32) + conv state for 48 GDN layers; each step verifies
  1 + 2 draft tokens per request (M = 48 rows) and drafts 2 tokens (2 MTP steps at M = 16).
- About every 7th step is a **mixed step**: one new request's prompt (~100-1900 tokens) +
  the other ~15 requests' 45 decode rows (vLLM chunked prefill, budget 2048 tokens?).

## Contract (what counts as the same computation)
task_definition.md: same function up to fp reordering; FP8 weights + FP8 KV fixed; activations
bf16 **or** the checkpoint's own W8A8 (dynamic per-token-group-128 e4m3, fp32 across K blocks)
per layer / per batch size; communication at baseline precision (bf16 all-reduce); exact
speculative decoding only; draft-side: FP8 draft lm_head allowed, reduced-vocab draft lm_head
allowed (non-benchmark frequency data), retrained draft heads allowed (non-benchmark data);
prefix caching stays on; parsers on; max-num-seqs >= 16; memory changes need the 128K test;
nothing tuned on the benchmark data.

## Starting point: the champion's design (BASE, 635.9 tok/s)
BASE = vLLM 4527515cd7 unmodified, serve args: TP=2, max-model-len 131072, max-num-seqs 16,
fp8 KV (5 GiB), MTP with 2 draft tokens, prefix caching on, chunked prefill on.
As read from the code and its server log:
- Linear layers: `MarlinFP8ScaledMMLinearKernel` (W8A16: FP8 weights dequantized to bf16 in the
  kernel, bf16 tensor cores). Marlin is picked before Triton block-W8A8 in
  `_POSSIBLE_FP8_BLOCK_KERNELS` (the cutlass/DeepGEMM/FlashInfer block kernels need sm_90+).
- Communication: NCCL all-reduce (custom all-reduce unavailable without P2P), `RING_LL` kernels
  with 2 CTAs (= 2 channels): 2 per layer + embedding = ~135 per target forward; lm_head is
  vocab-parallel and its logits are **all-gathered** (NCCL AllGather) for target and both
  draft steps.
- Compilation mode NONE (no torch.compile); CUDA graphs: FULL decode graphs up to 16 requests,
  piecewise/FULL "prefill" graphs only up to **96 tokens**; breakable CUDA graph enabled. Mixed
  steps above 96 tokens run eagerly (GDN chunk kernels launched one by one from Python).
- GDN decode: `gdn_decode_post_conv_mtp_kernel` with fp32 state (reads + writes 16 req x 24 heads x
  128x128x4 B = 25 MB per layer per GPU per step).
- Attention: FlashInfer paged prefill kernel (also used for the 3-row verify), fp8 KV.
- Sampling: target logits all-gathered, rejection sampler (greedy), draft argmax after gather.
- MTP drafter: 1 layer + the shared bf16 lm_head (cutlass wmma 16x16 kernel, grid 8 CTAs).
