# T1 map proposal 1 (leader-1): initial skeleton of map.md
map.md is empty; this proposes its whole initial content. Apply as given (replace the empty file).
Evidence: reports/T1-sweep/ (files 1-6), GPU job 1 (T1 worker NCCL/PCIe-IPC microbench,
agents/leader-1.w1/ar/), BASE profile task/baseline/nsys_c16.

---
# Map of the search space (Qwen3.8-27B-FP8, 2x RTX 4090, TP=2, c=16)

## 1. What the space is
**Time split (BASE):** decode steps 72.5% of benchmark time (38.5 ms each, 16 req x 3 rows), mixed
(prefill + decode) steps 27.5% (~148 ms each, ~330 prompt + 45 decode rows). Overall gain of a change ≈
0.725·ΔD/38.5 + 0.275·ΔP/148. Tokens per decode step = 16 x acceptance (2.345 at 2 drafts).
**Decode step parts (per GPU, BASE):** Marlin FP8 GEMMs 18.6 ms (12.2 GB at 0.66 of 1 TB/s) · all-reduce
7.7 ms (135 x ~58 us, NCCL via host SHM) · lm_head bf16 1.27 GB read 3x/step (target + 2 drafts) 4.2 ms ·
logits all-gathers 1.4 ms · GDN state update 3.0 ms (fp32 state r+w 2.1 GB) · MTP layer x2 ~0.9 ms ·
small kernels ~1.5 ms · idle 1.3 ms. Napkin at spec rates ~24 ms.
**Mixed step parts:** Marlin 81 ms (compute-bound, ~113 TFLOP/s bf16) · all-reduce 39 ms (294 us each at
375 rows) · GDN ~10 ms · rest ~18 ms.
**Pricing rules (calibrated):** NCCL AR(n rows of 5120 bf16) ≈ 53 us at n=42, +0.72 us/row (~7.3 GB/s payload);
Marlin ≈ bytes / 0.66 TB/s at M≤64 in-situ (0.8 isolated per ncu); bf16 GEMV lm_head ≈ 0.9 of BW; launch
floor ~3 us/kernel. Link: no GPU P2P; all traffic via host memory over PCIe 4.0.
**Dimensions / knob sites:**
- Combine nodes (128 all-reduces/forward + drafter's): implementation, overlap with GEMMs (split d), payload.
- Body GEMMs: kernel efficiency at M=42-64 (decode, BW-bound); FP8 W8A8 for mixed steps (compute-bound).
- Speculation: draft depth, drafter cost (draft lm_head precision/vocab, logits gather), tree vs chain.
- lm_head / sampling: logits all-gather vs local argmax (greedy), target and draft side.
- GDN state (fp32, in-place): kernel efficiency, write-back frequency (ReplaySSM).
- CPU/launch: idle between graphs, eager mixed steps, draft metadata rebuild.
- Parallelism: TP vs PP vs hybrid (see verdicts).

## 2. What we know, per part
| part | status | evidence |
|---|---|---|
| NCCL env tuning (PROTO Simple/LL/LL128, SHM_USE_CUDA_MEMCPY, NTHREADS/MIN_NCHANNELS) | **tested, no gain**: default already Simple at n≥42; LL/LL128 20-100% worse; others within noise | T1 job 1 (microbench) |
| FlashInfer PCIe-IPC all-reduce (env VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC) | **impossible here**: needs CUDA peer access ("peer access is not supported") | T1 job 1 |
| vLLM custom all-reduce | **impossible here** (needs P2P; disabled in BASE log) | BASE server log |
| zero-copy AR through mapped pinned host memory | untried (T1 testing) — napkin ~+10% | T1 6-ranking #1 |
| AR/GEMM overlap (split row-parallel output along d; RMSNorm scale commutes past next GEMM) | untried — napkin ~+9% | T1 4-knobs W1 |
| FP8 draft lm_head (`draft_lm_head_quantization: fp8`, ruling 1) | untried (T1 testing) | engine option |
| local argmax for draft tokens (`use_local_argmax_reduction`) | untried (T1 testing) | engine option |
| local argmax for target verification (greedy, generic fallback) | untried — ~+2% | T1 4-knobs R1 |
| reduced-vocab draft lm_head (ruling 2) | untried — ~+5% with fp8 | T1 4-knobs R3b |
| draft depth 3-4 (chain) | untried — +4% now, +9-11% with a cheap drafter | T1 4-knobs S1 |
| tree speculation | not now: GDN state must fork per tree node (needs a tree-aware GDN verify kernel) | T1 4-knobs |
| W8A8 FP8 GEMMs in mixed steps | untried — ~+6.5%; kernel speed on sm_89 unknown | T1 4-knobs R5 |
| Marlin decode efficiency (0.66 in-situ vs 0.8 isolated) | untried — ~+6% | T1 3-calibration |
| PP=2 / asymmetric PP | **not worth doing**: at decode each GPU must stream its 12.2 GB serially (~48 ms/step vs 38.5) and microbatches re-read weights; AR saved (7.7 ms) < cost | T1 4-knobs (napkin) |
| sequence parallel (RS+AG) | not worth doing: same bytes, 2x latency terms | napkin |
| lm_head replication | not worth doing: doubles lm_head bytes per GPU, costs KV budget | napkin |
| draft-in-verify bubble / decode microbatch split | not worth doing at decode: Shared-Read Rule doubles weight reads at N=48 ≪ ridge | napkin |
| mixed-step 2-microbatch AR overlap | untried — up to +7%, shrinks if AR gets faster | T1 4-knobs |
| bf16 GDN state, quantized comms | **forbidden** (lossy) | task_definition |
| ReplaySSM (GDN state write every L steps) | untried — ~+2%, high effort | T1 4-knobs K1 |
