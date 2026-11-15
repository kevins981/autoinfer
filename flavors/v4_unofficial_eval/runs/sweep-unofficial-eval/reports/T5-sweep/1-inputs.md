# T5 sweep, step 1: the framework's inputs for this task

Sources: config.json (Qwen/Qwen3.8-27B-FP8), task/TASK.md, task/SPEED_OF_LIGHT.md,
task_definition.md, assumption_ledger.md, HEAD = T3-004 (bank/T3-004/change.diff, serve_args.txt,
its cached unofficial starts bank/T3-004/unofficial/1/unofficial.json).

## 1. Model (config.json, text part)
| item | value |
|---|---|
| layers L | 64 = 48 GDN (gated delta-net, linear attention) + 16 full attention (every 4th layer: l = 3, 7, ..., 63) |
| hidden d | 5120; RMSNorm eps 1e-6 |
| MLP | SwiGLU, m = 17408 (gate_up 5120 -> 34816, down 17408 -> 5120), every layer |
| full attention | 24 q heads, 4 kv heads (GQA 6:1), head_dim 256, partial RoPE (25% = 64 dims), sigmoid/swish **output gate** (q_proj emits q and gate: 12288), q/k norm |
| GDN | 16 k heads x 128, 48 v heads x 128 (3 v heads per k head), conv1d width 4 on q,k,v, gates a (decay), b (beta), output gate z (6144), gated RMSNorm; recurrent state S[r,l,h] 128x128 **fp32** (`mamba_ssm_dtype float32`) |
| vocab V | 248,320, untied embedding and lm_head (bf16 each) |
| MTP | 1 extra layer (fc 2d -> d, one full-attention decoder layer + MLP), reuses the target lm_head |
| weights | FP8 e4m3, 128x128 block scales, activation_scheme dynamic (the checkpoint's W8A8 scheme) |

Weight sizes (FP8, total -> per GPU at TP=2):
| block | params | per GPU |
|---|---|---|
| MLP x64 | 267.4 M/layer, 17.1 B | 8.56 GB |
| GDN projections x48 (in_proj_qkvz 5120x16384, in_proj_ba 5120x96, out_proj 6144x5120) | 115.5 M/layer, 5.54 B | 2.77 GB |
| attention projections x16 (qkv+gate 5120x14336, o 6144x5120) | 105 M/layer, 1.68 B | 0.84 GB |
| **body** | 24.3 B | **12.2 GB** |
| lm_head (bf16, vocab-parallel) | 1.27 B | 1.27 GB |
| MTP layer (FP8) | ~0.37 B | ~0.19 GB |
| draft head in HEAD: FP8 copy of lm_head rows [0, 98304) | 0.50 B | 0.25 GB per draft argmax |

## 2. Hardware
2x RTX 4090 (sm_89, 128 SMs, 24 GB each, 72 MB L2). Per GPU:
| resource | value | note |
|---|---|---|
| DRAM BW | 1008 GB/s spec; ~0.9 TB/s achievable | calibrate |
| BF16 tensor, fp32 accumulate | 165 TFLOP/s dense | Ada whitepaper |
| FP8 tensor, fp32 accumulate | 330 TFLOP/s dense per whitepaper (SPEED_OF_LIGHT says 165: **check**) | calibrate with a microbench if prefill matters |
| GPU-GPU link | none direct: PCIe 4.0 x16 to host, ~25 GB/s/direction; HEAD's small all-reduce goes through pinned host memory (/dev/shm) | latency, not bandwidth, dominates for decode-sized messages |
| CPU | 48 cores, shared server | host-side scheduling runs here |

Ridge point (framework 01 §5): W8A16 (bf16 compute, 1-byte weights): N_ridge = 1 x 165e12 / (2 x 1e12) ~ 80 rows; W8A8 at 330 TF: ~165 rows.

## 3. Workload and objective
- ShareGPT, 400 prompts (unofficial evals: 200), prompt ~260 tokens incl. template, output = reference length, mean ~228; ~91K output tokens.
- Closed loop, 16 concurrent, temperature 0, chat completions; prefix caching on (ruling).
- Metric: output tok/s; gates: p99 TTFT/TPOT within 20% of BASE, GSM8K-250 >= 89.6, 128K request.
- HEAD's measured (unofficial, 200 prompts): 863.0 / 874.5 tok/s vs T1-019 ref 850.8; p99 TTFT 900 ms, p99 TPOT 37.9 ms; MTP acceptance 2.337 (mean tokens per step per request incl. bonus, I read it as).
- Implication: 91K tokens / ~870 tok/s ~ 105 s; at 16 x 2.34 = 37.4 tokens per full decode step that is ~2,450 steps -> ~43 ms of wall time per decode step all-in (decode + amortised prefill + tail with < 16 requests). The SOL floor is ~15 ms per step: the gap is the subject of the calibration (step 3).

## 4. Contract (what counts as the same computation)
- Exact up to fp reordering; FP8 weights + FP8 KV; activations bf16 or the checkpoint's W8A8 (dynamic per-token-group-128, fp32 across K-blocks), with a GSM8K A/B within 2 points.
- Communication at baseline precision (bf16 all-reduce; no quantised comm).
- Speculation exact (greedy verification against the target). Draft side is free (FP8 draft head, reduced-vocab draft head, retrained draft heads from non-benchmark data).
- Prefix caching stays on; max-num-seqs >= 16; 128K request must work; nothing fitted to the benchmark; tuning scheduler knobs only on non-benchmark prompts.
- Approximate rows of the framework (sparse attention, early exit, lossy KV) are **out** by contract: I list them only to mark them illegal.

## 5. Starting point: HEAD's design (T3-004, read from its diff)
| part | HEAD's choice |
|---|---|
| parallelism | TP=2 over everything (attention by heads, GDN by heads, MLP by m, vocab-parallel embedding / lm_head) |
| weight GEMMs | decode (M < 80): custom Triton **W8A16** on the checkpoint layout (no Marlin repack), per-(N,K,M) config table, for out_proj, gate_up, down, in_proj_qkvz, qkv; M >= 80: **W8A8** (per-token-group fp8 activations, vLLM triton block-scaled kernel); Marlin mid-batch config tweak remains for others |
| all-reduce | 2 per layer (after attention/GDN out_proj, after MLP down) = 128 per forward + MTP; decode-sized (<= 1.25 MB) via a **host-pinned shared-memory all-reduce** kernel (bitwise equal to NCCL sum); big ones NCCL; prefill: row-chunked GEMM + NCCL all-reduce overlap on a side stream (eager, M >= 192) |
| GDN decode | fused CUDA decode kernel (conv-post, delta rule, gated norm) for MTP rows; **replay mode with lazy commit**: state S written to HBM once every 2 steps; pending k/v/decay/beta rows kept in a dead state page and replayed (ReplaySSM/DeltaLog row B of framework 02 §4.3) |
| full attention | FlashInfer, FP8 KV; decode metadata planned from CPU upper bounds without a blocking seq_lens sync |
| speculation | MTP, 2 draft tokens, chain; FP8 draft lm_head; drafter argmax over token ids [0, 98304) only; target uses vocab-parallel local argmax (no full logits gather) + Triton greedy verify; batch-sharded sampling |
| graphs | CUDA graphs for decode; prefill/mixed cudagraph capture up to 512 tokens |
| scheduler | vLLM V2 model runner, chunked prefill, async scheduling (defaults), max 16 seqs, KV budget 5 GiB fp8, max_model_len 131072 |
