# T2 sweep, step 1: the framework's inputs for this task

Sources: `config.json` of Qwen/Qwen3.8-27B-FP8 (hf_hub_download), task/TASK.md, task_definition.md,
assumption_ledger.md, task/SPEED_OF_LIGHT.md, the BASE server log/config
(task/baseline/nsys_c16/server.log) and the baseline source tree.

## 1. Model (Input 1)
| | value | consequence |
|---|---|---|
| layers | 64 = 48 GDN (linear attention, `linear_attention`) + 16 full attention (every 4th) | two layer templates |
| hidden d | 5120 | activation row = 10 KB bf16 |
| MLP m | 17408 (SwiGLU: gate_up 5120x34816, down 17408x5120) | 133.7 MB fp8 per layer per GPU at TP=2 |
| full attention | 24 q heads, 4 kv heads, head_dim 256, output gate (q_proj also emits a gate: 24x256x2), partial RoPE 25%, qk-norm | qkv_proj 5120x14336, o_proj 6144x5120 |
| GDN | 16 key heads x128, 48 value heads x128, conv kernel 4, state 48x128x128 **fp32** (`mamba_ssm_dtype`) | in_proj_qkvz 5120x16384 (fp8), in_proj_ba 5120x96 (bf16), out_proj 6144x5120 |
| vocab | 248,320, untied lm_head bf16 (1.27 GB/GPU at TP=2), embedding vocab-parallel | lm_head = 3 layers' worth of bytes |
| MTP | 1 extra layer (full attention + MLP, fp8) + fc 10240->5120 (bf16) + the **shared full bf16 lm_head** | drafter reads the 1.27 GB lm_head per draft step |
| weights | FP8 e4m3, 128x128 block scales (`weight_block_size [128,128]`, activation_scheme dynamic) | |

Per GPU (TP=2) weight bytes per forward, from the shapes: GDN layer 43.0 (in_proj_qkvz) + 15.7 (out)
+ 89.1 (gate_up) + 44.6 (down) = 192.5 MB x 48 = 9.24 GB; attention layer 36.7 (qkv) + 15.7 (o) +
133.7 = 186 MB x 16 = 2.98 GB. **Body 12.2 GB**, target lm_head 1.27 GB, MTP layer ~0.2 GB + fc 0.05 GB.
Matches ncu: Marlin reads 43.2 / 16.3 / 91.2 / 46.2 MB per GDN-layer GEMM (task/baseline/ncu_marlin).

## 2. Hardware (Input 2)
Per GPU (RTX 4090, sm_89): DRAM BW ~1.0 TB/s (spec 1008 GB/s); L2 72 MB; 24 GB; tensor peak dense:
bf16 with fp32 accumulate ~165 TFLOP/s, FP8 with fp32 accumulate ~330 TFLOP/s (Ada whitepaper; the
SOL file says 165 for FP8/fp32-acc: to check by microbenchmark). Ridge: weight-only FP8 + bf16 MMA:
N_ridge = 1 B x 165e12 / (2 x 1e12) ~ 80 rows; W8A8: ~165 rows.
Link: **no P2P**; GPU0 <-> GPU1 through host memory (topology NODE, same NUMA node), PCIe 4.0 x16
~25 GB/s per direction (to measure). NCCL 2.30.7, ring LL. A missing resource in the framework's
hardware vector here: **CPU launch rate / host-side step time** (see 3-calibration).

## 3. Workload and objective (Input 3)
ShareGPT, 400 prompts (input ~260 tokens incl. template, output ~228), closed loop at concurrency 16,
temperature 0, chat completions. Score = output tok/s (BASE official 635.9). Gates: TTFT p99 / TPOT
p99 not >20% worse than BASE (1341 ms / 46.1 ms), GSM8K-250 >= 89.6%, one 128K request served.
Snapshot in steady state: 16 requests, each with ~260 + k tokens kept (KV for 16 attention layers;
GDN state + conv state for 48 layers); about every 6 decode steps one request finishes and a new
260-token prompt arrives (400 prompts over ~2,450 decode steps).

## 4. Contract (what counts as the same computation)
Exact up to floating-point reordering; FP8 weights and FP8 KV fixed; activations bf16 **or** the
checkpoint's W8A8 (dynamic per-token-group-128 e4m3 activations, fp32 accumulation across K blocks),
per layer / per batch size; communication at baseline precision (bf16); exact speculative decoding only
(any drafter, any draft count; verification exact); draft-side FP8 lm_head and reduced-vocab draft
lm_head (vocab chosen from non-benchmark data) allowed; nothing fitted to the benchmark;
max-num-seqs >= 16; 128K request must fit; parsers stay on.
So `approx` rows of the framework are out of scope except where the contract names them
(W8A8 activations, draft-side approximations whose verify stays exact).

## 5. Starting point: the champion's design (BASE = the task baseline)
- TP=2 over everything (attention heads, GDN heads, MLP m, vocab-parallel embedding and lm_head);
  2 all-reduces per layer (after o_proj/out_proj and after down), bf16, PYNCCL (custom all-reduce
  disabled: no P2P; FlashInfer AR "not supported for world_size=2").
- Linear layers: Marlin weight-only FP8 (bf16 MMA) for every M; small bf16 GEMMs (in_proj_ba, MTP fc,
  lm_heads) via cuBLAS/CUTLASS.
- Keep: FP8 paged KV (5 GiB budget) for 16 attention layers; GDN fp32 state + conv state per request;
  the MTP verify kernel writes the GDN state for **every** speculative position (3 slots per request),
  and the next step reads the slot of the last accepted token (rollback by slot choice).
- Speculate: MTP, chain of 2 drafts; draft step 1 runs on all 48 verified rows, step 2 on 16 rows;
  each draft step reads the full bf16 lm_head and all-gathers vocab-parallel logits.
- Window / batching: continuous batching, chunked prefill (max_num_scheduled_tokens 2048), prefix
  caching on; **compilation mode NONE** (no torch.compile); CUDA graphs via "breakable cudagraph",
  FULL decode graphs for 1..16 requests x 3 tokens, prefill/mixed graphs **only up to 96 tokens**:
  any step with a prompt chunk plus 48 decode rows above 96 tokens runs eager.
- Sampling: greedy (temp 0); target logits all-gathered across ranks (11.9 MB per GPU per step) before
  the rejection sampler.
